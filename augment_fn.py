import torch
from torch.utils.data import DataLoader, Dataset
import random
import nltk
from nltk.corpus import wordnet
import torchvision.transforms as transforms

# 確保下載 WordNet 數據（如果尚未下載）
nltk.download('wordnet')

class _AugmentedDataset(Dataset):
    def __init__(self, original_dataset, data_type, **kwarg):
        self.original_dataset = original_dataset
        self.new_dataset = []
        self._make_augument(data_type, kwarg)
        
    def _make_augument(self, data_type, kwarg): 
        transform_fn = _vision_transform(**kwarg)            
        for data, label in self.original_dataset:
            aug_data_1, aug_data_2 = transform_fn.apply(data)
            self.new_dataset.append((aug_data_1, label))
            self.new_dataset.append((aug_data_2, label))
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, idx):
        data, label = self.new_dataset[idx]
        return data, label

class _vision_transform:
    def __init__(self, **kwarg):
        self.image_size = kwarg["image_size"]
        self.mean = kwarg["mean"]
        self.std = kwarg["std"]
    
        self.strong_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.weak_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
        ])
        
    def apply(self, image):
        strong_image = self.strong_transform(image)
        weak_image = self.weak_transform(image)
        return strong_image, weak_image

def _nlp_transform(original_x, original_y, synonyms, p):
        EDAS = {
            'synonym_replacement': {'fn': _synonym_replacement, 'params': {'p': p, 'synonyms': synonyms}},
            'random_insertion': {'fn': _random_insertion, 'params': {'p': p, 'synonyms': synonyms}},
            'random_swap': {'fn': _random_swap, 'params': {'p': p}},
            'random_deletion': {'fn': _random_deletion, 'params': {'p': p}}
        }

        augmented_x = []
        augmented_y = []
        for sentence, y in zip(original_x, original_y):
            used_EDA_num = 2
            chosen_augmentations = random.sample(list(EDAS.values()), used_EDA_num)

            for i in range(len(chosen_augmentations)):
                augment_fn = chosen_augmentations[i]['fn']
                augment_params = chosen_augmentations[i]['params']
                augmented_x.append(augment_fn(sentence, **augment_params))
                augmented_y.append(y)
            
        return augmented_x, augmented_y
        
def _extract_unique_words(dataset):
    unique_words = set()
    for sentence in dataset:
        words = sentence.split()
        unique_words.update(words)
    return unique_words
    
def _get_synonyms(dataset):
    unique_words = _extract_unique_words(dataset)
    synonym_dict = {}
    for word in unique_words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym_list = []
            for syn in synonyms:
                for lemma in syn.lemmas():
                    synonym_list.append(lemma.name())
            synonym_dict[word] = list(set(synonym_list))
    return synonym_dict

def _synonym_replacement(sentence, p, synonyms):
    words = sentence.split()
    n = max(1, int(p * len(words)))
    eligible_words = [word for word in words if word in synonyms]  # 篩選出存在同義詞的單詞

    if len(eligible_words) == 0:
        return sentence  # 如果沒有可替換的單詞，直接返回原句子

    # 隨機選擇 n 個單詞進行替換，如果 n 大於可替換的單詞數量，則選擇全部可替換單詞
    words_to_replace = random.sample(eligible_words, min(n, len(eligible_words)))
    
    new_words = words.copy()    
    # 替換選定的單詞
    new_words = []
    for word in words:
        if word in words_to_replace:
            synonym_list = synonyms[word]  # 取出該單詞對應的同義詞列表
            new_word = random.choice(synonym_list)  # 隨機選擇一個同義詞
            new_words.append(new_word)  # 用同義詞替換
        else:
            new_words.append(word)  # 保留原單詞

    return ' '.join(new_words)  # 將列表中的單詞重新組合為句子

def _random_insertion(sentence, p, synonyms):
    words = sentence.split()
    n = max(1, int(p * len(words)))
    eligible_words = [word for word in words if word in synonyms]  # 篩選出存在於同義詞字典中的單詞

    if len(eligible_words) == 0:
        return sentence  # 如果沒有可插入的單詞，直接返回原句子

    # 隨機選擇 n 個單詞表示要做隨機位置插入同義字，如果 n 大於可選擇的單詞數量，則選擇全部單詞
    words_to_insert = random.sample(eligible_words, min(n, len(eligible_words)))

    # 對每個被選擇的單詞，隨機抽取同義字並插入句子的隨機位置
    for word in words_to_insert:
        synonym_list = synonyms[word]
        random_synonym = random.choice(synonym_list)  # 隨機選擇一個同義詞
        random_idx = random.randint(0, len(words))  # 隨機選擇插入位置
        words.insert(random_idx, random_synonym)  # 在句子的隨機位置插入該同義詞 
    return ' '.join(words)

def _random_swap(sentence, p):
    words = sentence.split()
    n = max(1, int(p * len(words)))
    for _ in range(n):
        idx1 = random.randint(0, len(words)-1)
        idx2 = random.randint(0, len(words)-1)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def _random_deletion(sentence, p):
    words = sentence.split()
    if len(words) == 1:
        return sentence

    new_words = []
    for word in words:
        if random.uniform(0, 1) > p:
            new_words.append(word)

    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)
    
            
def Vision_augment(data_loader, image_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    augmented_dataset = _AugmentedDataset(data_loader.dataset, "Vision", image_size=image_size, mean=mean, std=std)
    augmented_data_loader = DataLoader(augmented_dataset, batch_size=2*data_loader.batch_size, shuffle=True)
    return augmented_data_loader

def NLP_augment(original_x, original_y, p=0.1):
    synonyms = _get_synonyms(original_x)
    augmented_x, augmented_y = _nlp_transform(original_x, original_y, synonyms, p)
    return augmented_x, augmented_y