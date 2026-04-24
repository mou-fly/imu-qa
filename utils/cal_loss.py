import os
from utils.cider import compute_cider
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
# from pycocoevalcap.cider.cider import Cider
# from torchmetrics.text.cider import CIDERScore
from modules.simple_tokenizer import SimpleTokenizer
from openai import OpenAI
import ast
import numpy as np

# from new import candidate, reference

tokenizer = SimpleTokenizer()

"""
return bleu1 bleu2 bleu3 bleu4 METEOR ROUGE-L CIDEr
"""
def score_metric(out, truth, lens, index=None, infer=None):
    loss = []
    if isinstance(out, torch.Tensor):
        out = out.to('cpu').numpy()
    else:
        out = np.array(out)
    truth = truth.to('cpu').numpy()
    for i in range(len(out)):
        l = lens[i]
        loss_item = []
        o = out[i][1:l-1]
        t = truth[i][1:l-1]
        o_ = tokenizer.decode(o)
        t_ = tokenizer.decode(t)
        candidate = o_
        reference_l = [t_]
        # reference_l = tokenizer.decode(t)
        # bleu
        weights_bleu1 = (1, 0, 0, 0)
        weights_bleu2 = (0.5, 0.5, 0, 0)
        weights_bleu3 = (0.3, 0.3, 0.3, 0)
        weights_bleu4 = (0.25, 0.25, 0.25, 0.25)
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu1))
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu2))
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu3))
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu4))
        # METEOR
        loss_item.append(meteor_score(reference_l,candidate))
        # ROUGE-L
        sentence1 = ' '.join(t_)
        sentence2 = ' '.join(candidate)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        loss_item.append(scorer.score(sentence1,sentence2)['rougeL'].precision)
        # CIDEr
        # loss_item.append(compute_cider_single(sentence1,sentence2))
        loss_item.append(compute_cider(sentence1,[sentence2]))
        loss.append(loss_item)

        if index:
            print("cap : ", sentence1)
            print("pred: ", sentence2)
            cap = "cap : " + sentence1
            pred = "pred: " + sentence2
            infer_folder = "C:\project\caption\infer"
            infer_folder = os.path.join(infer_folder, "transformer_1")
            infer_file = infer_folder + ".txt"
            with open(infer_file, 'a') as f:
                f.write(cap)
                f.write("\n")
                f.write(pred)
                f.write("\n")


        if index is not None:
            if i < 3 and index == 1:
                print("cap : ", sentence1)
                print("pred: ", sentence2)
    loss = np.array(loss)
    loss = np.mean(loss,axis=0)

    loss = torch.tensor(loss, dtype=torch.float32, device='cuda:0', requires_grad=True)
    return loss

def metric(out, truth, lens, T):
    loss = []
    mRMC = []
    client = OpenAI(api_key="sk-2706bc3cc3f44cf280055719025e1d01", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    if isinstance(out, torch.Tensor):
        out = out.to('cpu').numpy()
    else:
        out = np.array(out)
    truth = truth.to('cpu').numpy()
    window = T // 30
    cnt = len(out) - window + 1
    for i in range(cnt):
        loss_item = []
        o_ = []
        t_ = []
        for j in range(window):
            index = i + j
            l = lens[index]
            o = out[i][1:l - 1]
            t = truth[i][1:l-1]
            o = tokenizer.decode(o)
            t = tokenizer.decode(t)
            for word in o:
                o_.append(word)
            for word in t:
                t_.append(word)
        candidate = o_
        reference_l = [t_]
        sentence1 = ' '.join(t_)
        sentence2 = ' '.join(candidate)
        rmc = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system",
                 "content": 'I will provide you with two paragraphs and 27 questions. 对于每个问题，请分别就两段话判断对此问题的回答是否一致，若一致，此问题的答案为1，否则为0.'
                            '如果让你感到困惑，则答案也记为1'
                            '最后的返回结果不需要分析过程，返回一个列表，列表的是元素是0或1'},
                {"role": "user",
                 "content": f"paragraph1-{sentence1} paragraphs2-{sentence2}"
                            f"question: "
                            f"(1) Please answer the number of times the user has consumed water during this period?  "
                            f"(2) Please answer whether the user has been drinking water during this period?  "
                            f"(3) Please answer whether the user has taken medication during this period?  "
                            f"(4) Please answer the number of times the user has taken medication during this period?  "
                            f"(5) Please answer whether the user has been reading during this period?  "
                            f"(6) Please answer whether the user read a book before lying down during this period?  "
                            f"(7) Please answer the number of times the user watered the plants during this period?  "
                            f"(8) Please answer whether the user watered the plants during this period?  "
                            f"(9) Please answer whether the user has opened windows for ventilation during this period?  "
                            f"(10) Please answer whether the user was playing with their phone while lying in bed during this period?  "
                            f"(11) Please answer whether the user has been walking during this period?  "
                            f"(12) Please answer the number of times the user has stretched during this period?  "
                            f"(13) Please answer whether the user has eaten fruits during this period?  "
                            f"(14) Please answer the number of times the user has eaten fruits during this period?  "
                            f"(15) Please answer whether the user has wiped the table during this period?  "
                            f"(16) Please answer whether the user has thrown away garbage during this period?  "
                            f"(17) Please answer whether the user washed their hands before eating fruits during this period?  "
                            f"(18) Please answer whether the user washed their hands after eating fruits during this period?  "
                            f"(19) Please answer whether the user has washed their hands after littering during this period?  "
                            f"(20) Please answer whether the user has washed their hands after wiping the table during this period?  "
                            f"(21) Please answer how many times the user washed their hands during this period?  "
                            f"(22) Please answer whether the user operated the mouse during this period?  "
                            f"(23) Please answer whether the user operated the keyboard during this period?  "
                            f"(24) Please answer whether the user has opened the envelope during this period?  "
                            f"(25) Please answer whether the user has turned on the desk lamp during this period?  "
                            f"(26) Please answer whether the user answered the phone during this period?  "
                            f"(27) Please answer the number of times the user answered the phone during this period?"},
            ],
            stream=False,
            max_tokens=1024
        ).choices[0].message.content
        rmc = ast.literal_eval(rmc)
        rmc = np.array(rmc)
        mRMC.append(rmc.mean())
        weights_bleu1 = (1, 0, 0, 0)
        weights_bleu2 = (0.5, 0.5, 0, 0)
        weights_bleu3 = (0.3, 0.3, 0.3, 0)
        weights_bleu4 = (0.25, 0.25, 0.25, 0.25)
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu1))
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu2))
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu3))
        loss_item.append(sentence_bleu(reference_l,candidate,weights=weights_bleu4))
        # METEOR
        loss_item.append(meteor_score(reference_l,candidate))
        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        loss_item.append(scorer.score(sentence1,sentence2)['rougeL'].precision)
        # CIDEr
        # loss_item.append(compute_cider_single(sentence1,sentence2))
        loss_item.append(compute_cider(sentence1,[sentence2]))
        loss.append(loss_item)

    loss = np.array(loss)
    loss = np.mean(loss,axis=0)

    rmc = np.array(mRMC)
    rmc = np.mean(rmc,axis=0)

    loss = torch.tensor(loss, dtype=torch.float32, device='cuda:0', requires_grad=True)
    return loss, rmc


# def compute_cider_single(prediction: str, reference: str):
#     hyp = {'0': [prediction]}
#     ref = {'0': [reference]}
#
#     scorer = Cider()
#     score, _ = scorer.compute_score(ref, hyp)
#     return score
#
# if __name__ == '__main__':
#     reference = "A small cat is sitting comfortably on a soft mat in the sun"
#     prediction = "A cat is sitting on a mat in the sunlight"
#     print(compute_cider_single(reference, prediction))
# from cider.cider import Cider

refs = {
    '0': ['a cat sits on a mat'],
    '1': ['a dog plays with a ball'],
    '2': ['a person rides a bike'],
}
hyps = {
    '0': ['a cat on a mat'],
    '1': ['a dog plays ball'],
    '2': ['a person ride bike'],
}

# scorer = Cider()
# score, scores = scorer.compute_score(refs, hyps)
# print('平均 CIDEr:', score)










