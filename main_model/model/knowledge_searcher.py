import faiss
import torch
import numpy as np

class KnowledgeSearcher:
    def __init__(self, config, mode, gpu_ids = [0],text_feat_path = 'YOUR KNOWLEDGE PATH', video_feat_path='YOUR KNOWLEDGE PATH'):
        self.text_feat_path = text_feat_path
        self.video_feat_path = video_feat_path
        self.config = config
        self.gpu_ids = gpu_ids  
        if config.knowledge_mode == "raw_clip":
            text_feat_path = "YOUR KNOWLEDGE PATH"
            video_feat_path = "YOUR KNOWLEDGE PATH"
        if mode == "coarse":
            text_feat_path = config.knowledge_corase_grained_text_path
            video_feat_path = config.knowledge_corase_grained_video_path
        
        
        self.text_features, self.video_features = self.load_features(self.text_feat_path, self.video_feat_path)
        self.text_features = self.text_features.cpu().detach()
        self.video_features = self.video_features.cpu().detach()

        self.text_index = self.build_index_gpu(self.text_features.cpu().numpy())
        self.video_index = self.build_index_gpu(self.video_features.cpu().numpy())

    
    
    def load_features(self, text_feat_path, video_feat_path):
        print("load features from {} and {}".format(text_feat_path, video_feat_path))
        text_features_list = torch.load(text_feat_path)
        video_features_list = torch.load(video_feat_path)
        text_features = torch.cat([torch.from_numpy(np_array) for np_array in text_features_list], dim=0)
        video_features = torch.cat([torch.from_numpy(np_array) for np_array in video_features_list], dim=0)
        print("load features finished from {} and {}".format(text_feat_path, video_feat_path))
        return text_features, video_features

    def normalize_features(self, features):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / norms

    def build_index(self, features):
        print("build text index from features")
        features = self.normalize_features(features)
        dimension = features.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(features)
        return index

    def build_index_gpu(self, features):
        print("build text index from features")
        features = self.normalize_features(features)
        dimension = features.shape[1]
        cpu_index = faiss.IndexFlatIP(dimension)
        
        gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.config.local_rank, cpu_index)
        gpu_index.add(features)  
        print("build text index from features finished")
        return gpu_index
    
    def build_index_gpu_ids(self, features):
        print("build text index from features")
        features = self.normalize_features(features)
        dimension = features.shape[1]
        
        cpu_index = faiss.IndexFlatIP(dimension)
        sharded_index = faiss.IndexShards(dimension)
        
        for gpu_id in self.gpu_ids:
            gpu_resources = faiss.StandardGpuResources() 
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, cpu_index)
            sharded_index.add_shard(gpu_index)
        
        sharded_index.add(features)
        print("build text index from features finished")
        return sharded_index

    def build_index_gpu_all(self, features):
        print("build text index from features on all GPUs")
        features = self.normalize_features(features)
        dimension = features.shape[1]
        
        cpu_index = faiss.IndexFlatIP(dimension)
        sharded_index = faiss.IndexShards(dimension)

        num_gpus = faiss.get_num_gpus()
        
        for gpu_id in range(num_gpus):
            gpu_resources = faiss.StandardGpuResources() 
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, cpu_index)
            sharded_index.add_shard(gpu_index)
        
        sharded_index.add(features)
        print("build text index from features finished on all GPUs")
        return sharded_index
        
    def search_similar_features(self, query_features, topK=5):
        query_features = self.normalize_features(query_features)
        distances, indices = self.text_index.search(query_features, topK)

        similar_text_features = self.text_features[indices.ravel()]
        similar_video_features = self.video_features[indices.ravel()]

        similar_text_features = similar_text_features.view(indices.shape[0], topK, -1)
        similar_video_features = similar_video_features.view(indices.shape[0], topK, -1)
            
        return distances, indices, similar_text_features, similar_video_features
    
    def search_similar_features_video(self, query_features, topK=5):
        query_features = self.normalize_features(query_features)
        distances, indices = self.video_index.search(query_features, topK)

        similar_text_features = self.text_features[indices.ravel()]
        similar_video_features = self.video_features[indices.ravel()]

        similar_text_features = similar_text_features.view(indices.shape[0], topK, -1)
        similar_video_features = similar_video_features.view(indices.shape[0], topK, -1)
            
        return distances, indices, similar_text_features, similar_video_features

    def update_features_and_index(self, query_indices, momentum=0.05):
        for idx in query_indices.ravel():
            self.text_features[idx] = momentum * self.text_features[idx] + (1 - momentum) * self.text_features[idx]
            self.video_features[idx] = momentum * self.video_features[idx] + (1 - momentum) * self.video_features[idx]
        
        self.text_index = self.build_index(self.text_features.cpu().numpy())


