import torch
from torch.utils.data import DataLoader, Dataset
import random
from nltk.corpus import wordnet

class _AugmentedDataset(Dataset):
    def __init__(self, original_dataset, data_type, **kwargs):
        self.original_dataset = original_dataset
        self.augmented_dataset = []
        self._make_augmentation(data_type, kwargs)
        
    def _make_augmentation(self, data_type, kwargs): 
        transform_function = _vision_transform(**kwargs)            
        for data, label in self.original_dataset:
            augmented_data_1, augmented_data_2 = transform_function.apply(data)
            self.augmented_dataset.append((augmented_data_1, label))
            self.augmented_dataset.append((augmented_data_2, label))
    
    def __len__(self):
        return len(self.augmented_dataset)
    
    def __getitem__(self, idx):
        data, label = self.augmented_dataset[idx]
        return data, label

class _vision_transform:
    def __init__(self, **kwargs):
        self.image_size = kwargs["image_size"]
        self.mean = kwargs["mean"]
        self.std = kwargs["std"]
    
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

def _nlp_transform(original_sentences, original_labels, synonyms, probability):
        AUGMENTATION_METHODS = {
            'synonym_replacement': {'fn': _synonym_replacement, 'params': {'probability': probability, 'synonyms': synonyms}},
            'random_insertion': {'fn': _random_insertion, 'params': {'probability': probability, 'synonyms': synonyms}},
            'random_swap': {'fn': _random_swap, 'params': {'probability': probability}},
            'random_deletion': {'fn': _random_deletion, 'params': {'probability': probability}}
        }

        augmented_sentences = []
        augmented_labels = []
        for sentence, label in zip(original_sentences, original_labels):
            used_augmentation_num = 2
            chosen_augmentations = random.sample(list(AUGMENTATION_METHODS.values()), used_augmentation_num)

            for i in range(len(chosen_augmentations)):
                augment_fn = chosen_augmentations[i]['fn']
                augment_params = chosen_augmentations[i]['params']
                augmented_sentences.append(augment_fn(sentence, **augment_params))
                augmented_labels.append(label)
            
        return augmented_sentences, augmented_labels
        
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

def _synonym_replacement(sentence, probability, synonyms):
    words = sentence.split()
    num_words = max(1, int(probability * len(words)))
    eligible_words = [word for word in words if word in synonyms]

    if len(eligible_words) == 0:
        return sentence

    words_to_replace = random.sample(eligible_words, min(num_words, len(eligible_words)))
    
    new_words = words.copy()    
    new_words = []
    for word in words:
        if word in words_to_replace:
            synonym_list = synonyms[word]
            new_word = random.choice(synonym_list)
            new_words.append(new_word)
        else:
            new_words.append(word)

    return ' '.join(new_words)

def _random_insertion(sentence, probability, synonyms):
    words = sentence.split()
    num_words = max(1, int(probability * len(words)))
    eligible_words = [word for word in words if word in synonyms]

    if len(eligible_words) == 0:
        return sentence

    words_to_insert = random.sample(eligible_words, min(num_words, len(eligible_words)))

    for word in words_to_insert:
        synonym_list = synonyms[word]
        random_synonym = random.choice(synonym_list)
        random_idx = random.randint(0, len(words))
        words.insert(random_idx, random_synonym)
    return ' '.join(words)

def _random_swap(sentence, probability):
    words = sentence.split()
    num_words = max(1, int(probability * len(words)))
    for _ in range(num_words):
        idx1 = random.randint(0, len(words)-1)
        idx2 = random.randint(0, len(words)-1)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def _random_deletion(sentence, probability):
    words = sentence.split()
    if len(words) == 1:
        return sentence

    new_words = []
    for word in words:
        if random.uniform(0, 1) > probability:
            new_words.append(word)

    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)
        
            
def Vision_augment(data_loader, image_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    augmented_dataset = _AugmentedDataset(data_loader.dataset, "Vision", image_size=image_size, mean=mean, std=std)
    augmented_data_loader = DataLoader(augmented_dataset, batch_size=2*data_loader.batch_size, shuffle=True)
    return augmented_data_loader

def NLP_augment(original_sentences, original_labels, probability=0.1):
    synonyms = _get_synonyms(original_sentences)
    augmented_sentences, augmented_labels = _nlp_transform(original_sentences, original_labels, synonyms, probability)
    return augmented_sentences, augmented_labels