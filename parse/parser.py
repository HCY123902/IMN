import random

train_source = open("./train_dial.txt", "r")
valid_source = open("./valid_dial.txt", "r")
test_source = open("./test_dial.txt", "r")

train_context = open("./train.txt", "w")
valid_context = open("./valid.txt", "w")
test_context = open("./test.txt", "w")

responses = open("./responses.txt", "w")

vocab = open("./vocab.txt", "w")

train_lines = train_source.readlines()
valid_lines = valid_source.readlines()
test_lines = test_source.readlines()

train_dials = []
valid_dials = []
test_dials = []

for line in train_lines:
    dial = line.split("__eou__")[:-1]
    dial = [t.strip().lower() for t in dial]
    train_dials.append(dial)

for line in valid_lines:
    dial = line.split("__eou__")[:-1]
    dial = [t.strip().lower() for t in dial]
    valid_dials.append(dial)

for line in test_lines:
    dial = line.split("__eou__")[:-1]
    dial = [t.strip().lower() for t in dial]
    test_dials.append(dial)

train_context_dial = []
valid_context_dial = []
test_context_dial = []

response_text = []

for dial in train_dials:
    if len(dial) > 1:
        train_context_dial.append({"text": dial[:-1], "response": len(response_text)})
        response_text.append(dial[-1])

        if len(dial) >= 7:
            assert dial[5] != dial[-1]
            train_context_dial.append({"text": dial[:5], "response": len(response_text)})
            response_text.append(dial[5])

for dial in valid_dials:
    if len(dial) > 1:
        valid_context_dial.append({"text": dial[:-1], "response": len(response_text)})
        response_text.append(dial[-1])

        if len(dial) >= 7:
            assert dial[5] != dial[-1]
            valid_context_dial.append({"text": dial[:5], "response": len(response_text)})
            response_text.append(dial[5])

for dial in test_dials:
    if len(dial) > 1:
        test_context_dial.append({"text": dial[:-1], "response": len(response_text)})
        response_text.append(dial[-1])

        if len(dial) >= 7:
            assert dial[5] != dial[-1]
            test_context_dial.append({"text": dial[:5], "response": len(response_text)})
            response_text.append(dial[5])

random.shuffle(train_context_dial)
random.shuffle(valid_context_dial)
random.shuffle(test_context_dial)

sample_range = range(len(response_text))

for i, context_map in train_context_dial:
    negavtive_sample = random.sample(sample_range)[0]
    if negavtive_sample == context_map["response"]:
        negavtive_sample = (context_map["response"] + 1) % len(response_text)
    context = context_map["text"].join(" _eos_ ")
    train_context.write("{}\t{} _eos_ \t{}\t{}\n", i, context, context_map["response"], negavtive_sample)

for i, context_map in valid_context_dial:
    negavtive_sample = random.sample(sample_range, 9)
    for j, sample in enumerate(negavtive_sample):
        if sample == context_map["response"]:
            negavtive_sample[j] = (context_map["response"] + 1) % len(response_text)
    
    context = context_map["text"].join(" _eos_ ")
    negavtive_response = negavtive_sample.join("|")
    valid_context.write("{}\t{} _eos_ \t{}\t{}\n", i, context, context_map["response"], negavtive_response)

for i, context_map in test_context_dial:
    negavtive_sample = random.sample(sample_range, 9)
    for j, sample in enumerate(negavtive_sample):
        if sample == context_map["response"]:
            negavtive_sample[j] = (context_map["response"] + 1) % len(response_text)
    
    context = context_map["text"].join(" _eos_ ")
    negavtive_response = negavtive_sample.join("|")
    test_context.write("{}\t{} _eos_ \t{}\t{}\n", i, context, context_map["response"], negavtive_response)


train_context.close()
valid_context.close()
test_context.close()

print("There are {} reponses captured".format(len(response_text)))
for i, text in enumerate(response_text):
    responses.write("{}\t{}", i, text)
responses.close()

vocab_count = {}
vocab_dict = {}
vocab_dict['unk'] = 0

for dial in train_dials:
    for utt in dial:
        tokens = dial.split(' ')
        for token in tokens:
            if len(tokens) > 0:
                vocab_count[token] = vocab_count[token] + 1

for dial in valid_dials:
    for utt in dial:
        tokens = dial.split(' ')
        for token in tokens:
            if len(tokens) > 0:
                vocab_count[token] = vocab_count[token] + 1

for dial in test_dials:
    for utt in dial:
        tokens = dial.split(' ')
        for token in tokens:
            if len(tokens) > 0:
                vocab_count[token] = vocab_count[token] + 1
    
for token, count in vocab_count:
    if count > 1:
        vocab_dict[token] = len(vocab_dict)





train_source.close()
valid_source.close()
test_source.close()
