from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch

from main_model.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from main_model.dataloaders.data_dataloaders import DATALOADER_DICT
from main_model.models.modeling import KDProR, AllGather
from main_model.models.optimization import BertAdam
from main_model.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import torch.nn.functional as F 
from main_model.utils.comm import is_main_process, synchronize
from main_model.utils.logger import setup_logger
from main_model.utils.metric_logger import MetricLogger

allgather = AllGather.apply

global logger


def get_args(description='KDProR for Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local-rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"],
                        help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='wti', help="interaction type for retrieval.")
    parser.add_argument('--wti_arch', type=int, default=2, help="select a architecture for weight branch")

    parser.add_argument('--cdcr', type=int, default=3, help="channel decorrelation regularization")
    parser.add_argument('--cdcr_alpha1', type=float, default=1.0, help="coefficient 1")
    parser.add_argument('--cdcr_alpha2', type=float, default=0.06, help="coefficient 2")
    parser.add_argument('--cdcr_lambda', type=float, default=0.001, help="coefficient for cdcr")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--interpolation", default=0.2, type=float, help="interpolation ")     
    parser.add_argument("--ot_iter_num", default=5, type=int, help="OT iter number")     
    parser.add_argument("--epsilon", default=1, type=float, help="epsilon")      
    parser.add_argument("--knowledge_scale", default=0.2, type=float, help="epsilon")      
    parser.add_argument("--t2v_knowledge_scale_logit", default=0.3, type=float,help='knowledge_scale_logit')
    parser.add_argument("--v2t_knowledge_scale_logit", default=0.1, type=float,help='knowledge_scale_logit')
    parser.add_argument('--knowledge_mode', type=str, help="knowledge mode" )

    parser.add_argument("--clip_logits_scale", default=0.1, type=float, help="clip_logits_scale")
    parser.add_argument("--model_type", type=str, default="KDProR", help="method type")
    parser.add_argument("--lambda_1", type=float, default=0.2, help="lambda_1")
    parser.add_argument("--lambda_2", type=str, default=0.3, help="lambda_2")
    
    
    args = parser.parse_args()

    return args



def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = KDProR(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dt %dv", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dt %dv", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, val_dataloader, train_dataloader, train_sampler, tokenizer


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    lr = args.lr  # 0.0001
    coef_lr = args.coef_lr  # 0.001
    weight_decay = args.weight_decay  # 0.2
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, max_steps):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    meters = MetricLogger(delimiter="  ")
    end = time.time()
    logit_scale = 0
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask, video, video_mask, inds = batch
        loss = model(text_ids, text_mask, video, video_mask)

        if n_gpu > 1:
            # print(loss.shape)
            loss = loss.mean()  # mean() to average on multi-gpu.

        with torch.autograd.detect_anomaly():
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        optimizer.step()
        optimizer.zero_grad()

        # https://github.com/openai/CLIP/issues/46
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if global_step % log_step == 0 and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "logit_scale: {logit_scale:.2f}"
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, t_mask_list, v_mask_list, t_feat_list, v_feat_list, mini_batch=32):
    sim_matrix = []
    logger.info('[start] map to main gpu')
    try:
        batch_t_mask = torch.split(t_mask_list, mini_batch)
    except:
        batch_t_mask = t_mask_list
        print("batch_t_mask")

    try:
        batch_v_mask = torch.split(v_mask_list, mini_batch)
    except:         
        batch_v_mask = v_mask_list
        print("batch_v_mask")        
    try: 
        batch_t_feat = torch.split(t_feat_list, mini_batch)
    except:
        batch_t_feat = t_feat_list
        print("batch_t_feat")         
    try:
        batch_v_feat = torch.split(v_feat_list, mini_batch)
    except:
        batch_v_feat = v_feat_list
        print("batch_v_feat")

    logger.info('[finish] map to main gpu')
    with torch.no_grad():
        for idx1, (t_mask, t_feat) in enumerate(zip(batch_t_mask, batch_t_feat)):
            logger.info('batch_list_t [{}] / [{}]'.format(idx1, len(batch_t_mask)))
            each_row = []
            for idx2, (v_mask, v_feat) in enumerate(zip(batch_v_mask, batch_v_feat)):

                b1b2_logits, *_tmp = model.get_similarity_logits(t_feat, v_feat, t_mask, v_mask)
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits) 

            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    return sim_matrix

def sinkhorn_knopp(log_sim_matrix, n_iters=4, detach=False):
    if detach:
        log_sim_matrix = log_sim_matrix.detach()
    m = log_sim_matrix.max()
    _log_sim_matrix = log_sim_matrix - m
    sim_matrix = torch.exp(_log_sim_matrix)
    b = 1 / sim_matrix.sum(0)
    for _ in range(n_iters):
        a = 1 / (sim_matrix @ b)
        b = 1 / (a @ sim_matrix)
    a = a / (1e-6 + a.sum(dim=0, keepdim=True))     
    b = b / (1e-6 + b.sum(dim=0, keepdim=True))

    log_a = a.log()
    log_b = b.log() - m

    return F.log_softmax(log_a, dim=0), F.log_softmax(log_b, dim=0)




def eval_epoch(args, model, test_dataloader,train_dataloader, device):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if isinstance(test_dataloader, list) and hasattr(test_dataloader[0].dataset, 'multi_sentence_per_video') \
            and test_dataloader[0].dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader[0].dataset.cut_off_points
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]
        sentence_num_ = test_dataloader[0].dataset.get_text_len()
        video_num_ = test_dataloader[0].dataset.get_video_len()

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
# ----------------------------
    #  2. cache the train text features
    # ----------------------------

    batch_train_feat_t = []
    batch_train_feat_v = []  
    with torch.no_grad():   
        for bid, batch in enumerate(train_dataloader): 
            batch = tuple(t.to(device) for t in batch)

            text_ids, text_mask, video, video_mask, inds  =batch
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

            
            text_feat, video_feat = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
            text_feat = text_feat.mean(dim=1)
            video_feat = video_feat.mean(dim=1)

            text_feat = text_feat.cpu().detach().numpy()
            video_feat = video_feat.cpu().detach().numpy()
            
            batch_train_feat_t.append(text_feat)
            batch_train_feat_v.append(video_feat)

            print("{}/{}\r".format(bid, len(train_dataloader)), end="")

    torch.save(batch_train_feat_t, 'YOUR TEXT FEATURE PATH')
    torch.save(batch_train_feat_v, 'YOUR VIDEO FEATURE PATH')
    

def main():
    global logger
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, val_dataloader, train_dataloader, train_sampler, tokenizer = build_dataloader(args)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        optimizer, scheduler, model = prep_optimizer(args, model, max_steps, args.local_rank)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps)
            torch.cuda.empty_cache()
            R1 = eval_epoch(args, model, val_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="")

                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                               'best.pth')
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            synchronize()
        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        # test on the best checkpoint
        model = model.module
        if args.local_rank == 0:
            model.load_state_dict(torch.load('best.pth', map_location='cpu'), strict=False)
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)

        torch.cuda.empty_cache()
        eval_epoch(args, model, test_dataloader, args.device)
        synchronize()

    elif args.do_eval:
        if args.local_rank == 0:
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["gen"](args, tokenizer)
            # modify your model path here
            model_file=""
            if os.path.exists(model_file):
                model_state_dict = torch.load(model_file, map_location='cpu') 
                if args.local_rank == 0:
                    logger.info("Model loaded from %s", model_file)
                # Prepare model
                model.load_state_dict(model_state_dict, strict=False)
                model.to(args.device)

                eval_epoch(args, model, test_dataloader, train_dataloader, args.device)

def load_model(epoch, model, args, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu') 
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        model.load_state_dict(model_state_dict, strict=False)

        model.to(device)
    else:
        model = None
    return model
if __name__ == "__main__":
    main()
