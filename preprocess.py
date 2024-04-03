import os
import sentencepiece as spm

def train_sentencepiece_model(data_dir='data', ka_file='ka_untokenized.txt', en_file='en_untokenized.txt', model_dir='model1', vocab_size=143):
    # Path for the combined training file
    combined_file_path = os.path.join(data_dir, 'combined_ka_en.txt')
    
    # Read Georgian and English files into lists
    with open(os.path.join(data_dir, ka_file), 'r', encoding='utf-8') as ka_f:
        ka_lines = ka_f.readlines()
    with open(os.path.join(data_dir, en_file), 'r', encoding='utf-8') as en_f:
        en_lines = en_f.readlines()

    # Combine Georgian and English sentences into one for training
    with open(combined_file_path, 'w', encoding='utf-8') as combined_f:
        for ka_line, en_line in zip(ka_lines, en_lines):
            combined_f.write(ka_line.strip() + '\n' + en_line.strip() + '\n')
    
    # Train SentencePiece model
    model_prefix = os.path.join(model_dir, 'ka_en')
    spm.SentencePieceTrainer.train(input=combined_file_path, model_prefix=model_prefix, vocab_size=vocab_size, model_type='unigram')
    print("SentencePiece model training complete.")

def tokenize_georgian_word(word):
    suffixes = ['ის', 'ით', 'ად', 'მა', 'ი', 'მ', 'ს', 'ო', 'ვ']
    suffix_pattern = re.compile('(?:' + '|'.join(suffixes) + ')$')
    match = suffix_pattern.search(word)
    if match:
        suffix = match.group(0)
        root = word[:-len(suffix)]
    else:
        root = word
        suffix = ''
    return root, suffix

def tokenize_text_files(data_dir='data', model_dir='model1', ka_file='ka_untokenized.txt', en_file='en_untokenized.txt'):
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(model_dir, 'ka_en.model'))

    # Tokenization for Georgian text
    ka_input_path = os.path.join(data_dir, ka_file)
    ka_output_path = ka_input_path.replace('untokenized', 'tokenized')
    with open(ka_input_path, 'r', encoding='utf-8') as in_f, \
         open(ka_output_path, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            tokenized_line = []
            for word in line.strip().split():
                root, suffix = tokenize_georgian_word(word)
                # Optionally apply SentencePiece tokenization on root here
                tokenized_line.append(root + (f" -{suffix}" if suffix else ""))
            out_f.write(' '.join(tokenized_line) + '\n')
    print(f"Tokenization complete for {ka_file}. Output saved to {ka_output_path}.")

if __name__ == '__main__':
    data_directory = 'data'
    model_directory = 'model1'
    georgian_file = 'ka_untokenized.txt'
    english_file = 'en_untokenized.txt'
    
    os.makedirs(model_directory, exist_ok=True)
    
    train_sentencepiece_model(data_dir=data_directory, ka_file=georgian_file, en_file=english_file, model_dir=model_directory)
    tokenize_text_files(data_dir=data_directory, model_dir=model_directory, ka_file=georgian_file, en_file=english_file)