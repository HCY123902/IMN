import random
import json
import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize

train_source = open("./train_dial.json", "r")
valid_source = open("./valid_dial.json", "r")
test_source = open("./test_dial.json", "r")

train_context = open("./train.txt", "w")
valid_context = open("./valid.txt", "w")
test_context = open("./test.txt", "w")

responses = open("./responses.txt", "w")

vocab = open("./vocab.txt", "w")

train_lines = json.load(train_source)
valid_lines = json.load(valid_source)
test_lines = json.load(test_source)

train_dials = []
valid_dials = []
test_dials = []

for line in train_lines[:7000]:
    dial = ['{}: {}'.format(turn["speaker"], turn["utterance"]) for turn in line]
#     print(line)
    try:
        dial = [word_tokenize(t.strip().lower()) for t in dial]
        train_dials.append(dial)
    except UnicodeDecodeError:
        print("Detected unicode that is not parsable")
        continue

for line in valid_lines[:200]:
    dial = ['{}: {}'.format(turn["speaker"], turn["utterance"]) for turn in line]
    try:
        dial = [word_tokenize(t.strip().lower()) for t in dial]
        valid_dials.append(dial)
    except UnicodeDecodeError:
        print("Detected unicode that is not parsable")
        continue

for line in test_lines[:200]:
    dial = ['{}: {}'.format(turn["speaker"], turn["utterance"]) for turn in line]
    try:
        dial = [word_tokenize(t.strip().lower()) for t in dial]
        test_dials.append(dial)
    except UnicodeDecodeError:
        print("Detected unicode that is not parsable")
        continue

train_context_dial = []
valid_context_dial = []
test_context_dial = []

response_text = []

for dial in train_dials:
    if len(dial) > 1:
        train_context_dial.append({"text": dial[:-1], "response": len(response_text)})
        response_text.append(dial[-1])

        if len(dial) >= 7:
#             assert dial[5] != dial[-1]
            train_context_dial.append({"text": dial[:5], "response": len(response_text)})
            response_text.append(dial[5])

for dial in valid_dials:
    if len(dial) > 1:
        valid_context_dial.append({"text": dial[:-1], "response": len(response_text)})
        response_text.append(dial[-1])

        if len(dial) >= 7:
#             assert dial[5] != dial[-1]
            valid_context_dial.append({"text": dial[:5], "response": len(response_text)})
            response_text.append(dial[5])

for dial in test_dials:
    if len(dial) > 1:
        test_context_dial.append({"text": dial[:-1], "response": len(response_text)})
        response_text.append(dial[-1])

        if len(dial) >= 7:
#             assert dial[5] != dial[-1]
            test_context_dial.append({"text": dial[:5], "response": len(response_text)})
            response_text.append(dial[5])

random.shuffle(train_context_dial)
random.shuffle(valid_context_dial)
random.shuffle(test_context_dial)

sample_range = range(len(response_text))

for i, context_map in enumerate(train_context_dial):
    negative_sample = random.sample(sample_range, 1)[0]
    if negative_sample == context_map["response"]:
        negative_sample = (context_map["response"] + 1) % len(response_text)
    context = " _eos_ ".join([' '.join(tokens) for tokens in context_map["text"]])
    train_context.write("{}\t{} _eos_ \t{}\t{}\n".format(i, context, context_map["response"], negative_sample))

for i, context_map in enumerate(valid_context_dial):
    negative_sample = random.sample(sample_range, 9)
    for j, sample in enumerate(negative_sample):
        if sample == context_map["response"]:
            negative_sample[j] = (context_map["response"] + 1) % len(response_text)
    
    context = " _eos_ ".join([' '.join(tokens) for tokens in context_map["text"]])
    negative_sample = [str(n) for n in negative_sample]
    negative_response = "|".join(negative_sample)
    valid_context.write("{}\t{} _eos_ \t{}\t{}\n".format(i, context, context_map["response"], negative_response))

for i, context_map in enumerate(test_context_dial):
    negative_sample = random.sample(sample_range, 9)
    for j, sample in enumerate(negative_sample):
        if sample == context_map["response"]:
            negative_sample[j] = (context_map["response"] + 1) % len(response_text)
    
    context = " _eos_ ".join([' '.join(tokens) for tokens in context_map["text"]])
    negative_sample = [str(n) for n in negative_sample]
    negative_response = "|".join(negative_sample)
    test_context.write("{}\t{} _eos_ \t{}\t{}\n".format(i, context, context_map["response"], negative_response))


train_context.close()
valid_context.close()
test_context.close()

print("There are {} reponses captured".format(len(response_text)))
for i, tokens in enumerate(response_text):
    responses.write("{}\t{}\n".format(i, ' '.join(tokens)))
responses.close()

vocab_count = {}
vocab_dict = {}
vocab_dict['unk'] = 0

for dial in train_dials:
    for tokens in dial:
#         tokens = tokens.split(' ')
        for token in tokens:
            if len(tokens) > 0:
                vocab_count[token] = 1 if token not in vocab_count else vocab_count[token] + 1

for dial in valid_dials:
    for tokens in dial:
#         tokens = utt.split(' ')
        for token in tokens:
            if len(token) > 0:
#                 if len(token) > 1 and token[-1] in ['.', ',', ':', '?', ';', '!', '']:
#                     punctuation = token[-1]
#                     vocab_count[punctuation] = 1 if token not in vocab_count else vocab_count[token] + 1
#                     token = token[:-1]
                vocab_count[token] = 1 if token not in vocab_count else vocab_count[token] + 1

for dial in test_dials:
    for tokens in dial:
#         tokens = utt.split(' ')
        for token in tokens:
            if len(token) > 0:
                vocab_count[token] = 1 if token not in vocab_count else vocab_count[token] + 1
    
for token, count in vocab_count.items():
    if count > 1 and len(token) > 0:
        vocab_dict[token] = len(vocab_dict)

for token, position in vocab_dict.items():
    vocab.write("{}\t{}\n".format(token, position))

vocab.close()

train_source.close()
valid_source.close()
test_source.close()
