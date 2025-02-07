import torch
from torch import nn
import torch.nn.functional as F
from .utils_function.sinkhorn_knopp import sinkhorn_knopp
from icecream import ic  




class knnLoss(nn.Module):
    def __init__(self):
        super(knnLoss, self).__init__()


    def loss(self, knn_logits, targets):

        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        knn_loss = F.nll_loss(torch.clamp(torch.log(p), min=-100),
            targets, reduction="mean")

        return knn_loss

    def forward(
        self, loss, knn_logits,
        targets, coeff
    ):
        loss = self.loss(
            loss, knn_logits, targets,
            coeff)
        return loss


class knnFocalLikeLoss(nn.Module):
    def __init__(self):
        super(knnFocalLikeLoss, self).__init__()

    def is_single(self):
        return False

    def loss(self, logits, knn_logits, targets, gamma):
        loss = F.cross_entropy(logits, targets, reduction="none")

        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)
        # modulator = (1 - p_t) ** gamma
        # below is a numerically stable version
        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        p_t = torch.sum(p * targets, -1)
        # a mask of p == 0
        modulator = torch.exp(gamma * torch.log1p(-1 * p_t))

        loss = loss * modulator
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, knn_logits,
        targets, coeff
    ):
        loss = self.loss(
            pred_logits, knn_logits, targets, coeff)
        return loss
    
    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        return target
        
class KnowledgeStore(nn.Module):

    def __init__(self, K=8192, text_dim=512, word_num = 32, vid_dim=512, frame_num=12, knn=1, lamb=0.1, topK=3):
        super(KnowledgeStore, self).__init__()

        self.register_buffer("queue_text", torch.randn(K, word_num, text_dim))
        self.register_buffer("queue_video",torch.randn(K, frame_num, vid_dim))
        
        self.register_buffer("queue_ptr_t", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_v", torch.zeros(1, dtype=torch.long)) 
        self.knn = knn
        self.word_num = word_num
        self.frame_num = frame_num
        self.lamb = lamb
        self.K = K


    def get_size(self):
        return self.K
    
    def enqueue(self, texts, videos ):
        with torch.no_grad():
            ptr_t = int(self.queue_ptr_t)
            ptr_v = int(self.queue_ptr_v)
            bs_t = texts.shape[0]
            bs_v = videos.shape[0]


            if ptr_t + bs_t <= self.K:
                self.queue_text[ptr_t:ptr_t + bs_t, :] = texts
            else:
                part1 = self.K - ptr_t
                self.queue_text[ptr_t:self.K, :] = texts[:part1]
                part2 = bs_t - part1
                self.queue_text[0:part2, :] = texts[part1:]

            if ptr_v + bs_v <= self.K:
                self.queue_video[ptr_v:ptr_v + bs_v, :] = videos
            else:
                part1 = self.K - ptr_v
                self.queue_video[ptr_v:self.K, :] = videos[:part1]
                part2 = bs_v - part1
                self.queue_video[0:part2, :] = videos[part1:]
            if self.queue_text.shape[0] ==0 or self.queue_video.shape[0] ==0:
                print(20*" error ")
            self.queue_ptr_t[0] = (ptr_t + bs_t) % self.K
            self.queue_ptr_v[0] = (ptr_v + bs_v) % self.K


    def distributed_transmission_video(self, get_similarity_logits, video_feat ):
        # with torch.no_grad():
        batch_size_v = video_feat.shape[0]
        store_text_feat = self.queue_text.clone().detach()  
        batch_t_feat = torch.split(store_text_feat, batch_size_v) 
        create_text_mask = torch.ones(batch_size_v, self.word_num, requires_grad=False).to(device=video_feat.device)
        create_video_mask = torch.ones(batch_size_v, self.frame_num, requires_grad=False).to(device=video_feat.device)
        t2v_logits_list = []
        for text_feat in batch_t_feat:
            t2v_logits, v2t_logits, _ = get_similarity_logits(text_feat, video_feat, create_text_mask, create_video_mask) 
            t2v_logits_list.append(t2v_logits)
        t2v_logits = torch.stack(t2v_logits_list)
        t2v_logits = torch.mean(t2v_logits, dim=0)

        text_bias, video_bias = sinkhorn_knopp(t2v_logits)

        return video_bias

    def distributed_transmission_text(self, get_similarity_logits, text_feat ):

        batch_size_t = text_feat.shape[0]
        store_video_feat = self.queue_video.clone().detach() 
        batch_v_feat = torch.split(store_video_feat, batch_size_t) 
        create_text_mask = torch.ones(batch_size_t, self.word_num, requires_grad=False).to(device=text_feat.device)
        create_video_mask = torch.ones(batch_size_t, self.frame_num, requires_grad=False).to(device=text_feat.device)
        
        v2t_logits_list = []
        for video_feat in batch_v_feat:
            t2v_logits, v2t_logits, _ = get_similarity_logits(text_feat, video_feat, create_text_mask, create_video_mask) 
            v2t_logits_list.append(v2t_logits)
        v2t_logits = torch.stack(v2t_logits_list)
        v2t_logits = torch.mean(v2t_logits, dim=0)
        ic(v2t_logits)
        video_bias, text_bias  = sinkhorn_knopp(v2t_logits)

        return text_bias

    def knn_infer_text(self, query):

        # kl_div.shape = [1, len(self.queue_text)]
        kl_distance = torch.mean(self.queue_text[:, None, :] * (self.queue_text[:, None, :].log() - query.log()), dim=2).transpose(1, 0)
        if self.knn == 1:
            # directly return the nearest neighbor
            return self.queue_video[kl_distance.argmin(dim=1)].tolist()
        else:
            values, indices = torch.topk(kl_distance, self.knn, dim=1, largest=False)
            # count for each category within k nearest neighbors, and return the dominant category
            # knn_cnt.shape = [1, self.n_class]
            knn_cnt = torch.zeros((query.shape[0], self.n_class))
            for i in range(self.n_class):
                knn_cnt[:, i] = (self.queue_video[indices] == i).sum(dim=1)
            return knn_cnt.argmax(dim=1).tolist()
