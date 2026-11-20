"""
005_name-that-llm: Exp #016 Full 7-Backbone + Two-Way Gauss Head (Llama3-S1 + Qwen3-8B-Base) + Qwen3-Embedding-8B Clusters + Global Sinkhorn(0.30) + GLM-5.3/Flash
5-Fold OOF Balanced Accuracy: 0.75130 (Nested-CV: 0.75094, c_sk=0.00 removing Cluster Sinkhorn overfitting)
"""

import base64
import gc
import glob
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from safetensors.torch import load_file as load_safetensors
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags, hstack
from scipy.spatial.distance import squareform
from scipy.special import log_softmax, logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

MAX_LEN = 416
W_QWEN3_8B = 0.225
W_QWEN3_4B = 0.2025
W_MB_LARGE = 0.2025
W_DEB_LARGE = 0.135
W_QWEN3_17B = 0.10125
W_MB_BASE = 0.03375
W_LLAMA3_8B = 0.10
W_STYLEGRAM_V1 = 0.35
W_STYLEGRAM_V2 = 0.35
W_GAUSS = 1.8
QWEN_EMB_THR = 0.65
GAUSS_D = 256
GAUSS_TEMP = 3.0
FAMILY_GATE_LAM = 0.15
CLUSTER_SK_ALPHA = 0.0
GLOBAL_SK_MIX = 0.30
EPS = 1e-8

DEADLINE_H = 8.65
RESERVE_S = 180.0
MAX_COVER = 0.30
MIN_TEXTS = 50
MAX_CHANGED = 0.25
SEQ_BATCH = 8
PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>"
NTL_SEGRUN_SHA256 = "9c39e3afc77c64c1930f438fef50f297a6a8a649252ab4af9f618804be84cc8c"
FIXED_S_FLASH = 1050.0
ROW_S_FLASH = 0.0115
FIXED_S_GLM53 = 1500.0
ROW_S_GLM53 = 0.0205

FUNCTION_WORDS = frozenset(
    "a an the and or but if while although because as so than that which who whom whose "
    "this these those it its they them their theirs he him his she her hers we us our ours "
    "you your yours i me my mine in on at to from with without for of by over under between "
    "through into about is are was were be been being have has had do does did can could will "
    "would shall should may might must not no nor all any some each every both either neither "
    "other another such only also just very too more most less least here there when where why "
    "how what whether then now still yet even much many few little".split()
)

V1_CODE_B64 = "H4sIAFZd/GoC/+1c+3PbRpL2z/wr5pyrAuCQEEk9bDOm65K17Lg2mzixsrdVXB4LIkASSxKAAVCPKPrf7+vuGTxIUFZuL6mrW6ISGo959PT09HR/3SP3yD36jw/ezbeB5wfpk9/l6sq1799u9/i4vKf3vW6/13uibp78Adcmy70U3T/517z6z9U6D9fBsPf85clx//Tli67b7b98+fKs9eRw/f+/ZoGXb9Igc5Pb360PWtRnZ2f71v/xcb/7pHfaf35ydnz2/PhE1v/zJ6p7WP+/+/X06dMPQdrJg5tcrcJlsAoXcewrIxZqlsZrlS8ClVCpeBlEKpvG9CWeqShfdbJgnm4it9VCM/LJV1xuoFaJGqpVPO8kaXxJ5amdPN0EUqCtvsX3IMrTOLk1nyMQovvxwyxPw8tNHsZRu7Xuo/D5JBmhweS/+mNlZ7H6q5fqN2N8RZGO+va/+k5bpV60xJtFfK3WXnQrHWbqEsMqqYijQNldFMvjpNNz3BaY0QrXSZzmKtqsQZSXqSgxrxIv8vEC/yV+q9Xygxm1YGdtFU2mCy/NnEFL4VolbbVog5qCjGw0aKvu2PWy/DYJ7ChxZ6vYy89OQCl/6z3wrf/At+Mx9xgRn4PIXiUOP195Kd6g+Nq7Cdebtc2cWahnRFcv6LyUYgtvNSO2eTd2pI6OFAjuyZdPUv3TxovycBXYNKQRNoZTDMPtdem3z/enfP+c7192x44KIRUqWGUB1f8lSOPMPpM20wASFak7fqDLiiaYFmugMLvFu1UyWQdeZJH0uHRnO/WvWe7LR9xsf/vUPcW3T6PuuC3PvS4/98xzX773zfOpfD82z8/l+4l5finfT8dbFIaGwLBOH4TZkL/YpZ6+CvmLbeq/UG8xyZ03QR5M83cfLpQXeavbPJyq6Sa94rU4YGmeYWpZgLFOrIxXV2XZZmEOAb2MrwIutI79YIVS8TXm5CZB0x4tppKimT8HNZhe9aVaOG4GUXHUEU1d9inNbciRvKuOItukSRpmgRmo3UH1DlXXA1aqHFV8mQXpFVSCruWtFHi2yTQ51Q9lD78UTRvK6jQ5ToNkxHNabKYmSuNNL7HpZUNxrPgedSALdKi6zWVOizKv1GlzkX63Uqbf1A7TNQ963bLk66HCo9Mo4PRuMgvTLGcZGw1omY73F82CaRzJmhhR0YEpK2vxtSxzXpJNS+oymaJuhxaUnn3WB6LRSCEI78FNu18f+nKCPYHLEcN3K0rhe60qSWAnZluxZQ9p81ujN7WCSHz3jZd7b1NvHdgjrWFJu+WOo2YxNhlUU2GkfgmTrXbGzsFo/Y2X+3/C/zvZ9f/6B//vD/H/XtT8v5OXL0/d4xcvT16cvjispX+BC+Z7GP+ezt8j/L/nvTPj/53iEe+fn3ZPDv7fH+T//QTVD3vTV9dpmIvheBlGXnqrZrD9txw9uF2BlFnwjsF+WxyvsqOyTOXWnSaJ41b8Krh0m2neanKzxE5gIia8n9uJly/qNsJ1mC9UnMAakG/W9aXlUP3ZoDBNZi63YV9a3198d2E5Ox+EBjfxpkvbevXe0uYFd+NUisPWYEODv5Tt03VJXqMbRFNY2La1yWedF5WOPt/ZZbWjGtFOnQ/st+rBZsEnwwdwlJ5g9MFJpjmwQ58d0SQNZuGNsPx/xLA/P5phTM4Wv2pUEO+YytpIUUS8Sy/z0tS7Fcp98nCH1qvw5HF8NESgslMZd3NdFHLz+PI2h+XpGAankN8qfzVn400O8kbjZu6lu9zzMrg4OfqiBu0Th9wJzcmijB21HcIBZAybqMLJol6dkRNiHnyFOdx+p87AKMeAMd6mBistvthiRsF3gnQuN7NZkNqmcyADUe48MAvgiuslYIRva0kDAU7NsUeRKmfFMq9yFiL5nZZWhjGO+4oFIFOjSIr7bXUyHpBTS4hR22BDbXVuUB+BVArp/t+brA/lZAGYima/gbmfmS47yn7L5G9xe/98ZfQzq0zaDJOG79nCI6yI5mj28AwBDQiqE/RPMfBjyUAfAnrdFjlNGxn5/sdy8PCZnUpVjAVV7roDxSOCI8l3fevenQe57edlYfi3Uj7M1PdwFAeEOFzOemc1Zqbe9T6574OF18TGtMLGTR9s1IgbnsBU9eqV2m5UOIq23aswuLa3uZ+CAyWhunQjDXjJPdv867jQV+ss/CXYJU0XqPdRUEr9tw7+38H++yf8v5Oz56e9nvvy+Yvjbu/g//0rXLDRJ0mYAECOgt/LDXzY/zs+7Z8ds/9HN/3ntP578AAP/t8f5P99JNNLnBy1icin8+AAztW77/4iEQS4ewibZYsw4ejXyrtFLBD+3Rr2mXr37ue3sDXYdqjFA8+vAriQHFicx3Aj80Uab+YLZSouEY2bUc/Fiy97tJPDNPTJEkR4L/CmC3UdhPNFTl9ov0TIbcrBSDVdbGAJti7gimJLDP0NYgts0qDd/DqgQKU0jCbDKxAAC40ifuTU4gb9TFEDQcYljMqAHIUcdVAoWXnTwG29l4AkFVBTL4riHAGYlc8vpR8Yst5qJZxrF1EZ8IgGlaxCch7zWAjNlM2j4QceCXEkaAmDvbkXRghAssWQ3OYL0Le1MlWnA69cffP++8mb9z/hCaPLlPXuux++sfA0za4Q1EQrLt11OmQZ//DzhS6be3PY3F62KEySUadzHadLdZSvE/LXx/RGu20fL37iR+D5HBju6Ogp4vT8Hj5d59LLMZoX8kzc6Igp5m3yeFztZRWuwYjvuaCPYdNwOpn6KB3CIOSbDbfX8cMr9YFfXG58mHtlQVAWUI+V+443yyEHb8fjVkuagsykeVaZiUsd+VXTeAMJs1cxLHS4IBxecRj0oHpcZS2zlXg+xE/5Abn2mcAdW3jGVy2WAQmD02QnYPYl9U9gAT6FEb2JU6wlV1XGAhn2EI5JfSVcYYQB5a8hWAGEbLUaqICXTQVwodct1Nfjwl4Nx3eW00qkaB+9wGS3eTDCERY0s6qKSjrAntL4QQg8rH9IBK4I8Zv10vJpnWD9Idj4FVnZINWnJUxrYBbmIutEFxZCDKXAiz+4QTFimjpW9jc85PObaRCAm04JL3EjizCau61yJgeyqiuDXnhGEVT0hf0UkS5WKLR+bp/CMvaY73NolohWJmsIWVxVrUKtt2xM7pEfXB1Fm9XKYf1C/Eu8uZ5FPXn0AoOb8ryQRmB2rckf879S/iYF8VwQ6ykvugmFqKy1I5924M5drsBSV9SoJDyImnVcRcosQ9xLMVbRBuGrW0xnugqDdFCVuE2EMBtk2YfO+E/hq17tR3dY7Pfa/Xaj5Bdll6kb2rM+0o710bp/RP40OprNsoC0mOAp0kaRGwSV0qIZlNeQJpq/f2RxVEuZ8NJ5gtBfYJ7nq/jS3FNhcx9n5i4tymYLpHmstkBC87S5BNHTICvqZbfFrcw56CleQNZbj87iQEsuObtuGJFfayOdIdZv/DCNKARpnr3LjP61JxPaQSYTgr50m0W2DPzPKP7kDdT5SbdfUEsQ+9an1p++/fpi8uGn87fv/wbv1BrN//L1xz+PX2Vx8vrVrxvQ8iv+hbMNwAQ5GHigRbN8/epI/rW0I4+2JyH27JuqH79Gi2ngZhCc6cJOLRSy/+5/6XTiWYdvrHKUl14W8DC5vris2sdftxW9rCII0JD22qXlltg9ZAXUXvQLaG0+38wmy6vJ5rivoYRlcFuiQD/JPg4pRnWgQOsg93yEfJG3skJaDusjjw0LAzKTgqEdjRYMNc5aG+t0P8z5aOiCuimhiwm0b1tNGLpYXjVhFz9uQxcldkHACsrbOwBQI/7z436oTnPbIASOK7uRAZppeAFbKValewINMg2d9Bg2wW9/QFk9x/x7MlAnbXXKv2f8+1xKosIL/NOTf1DlxX19VNkSofZ8a1jYF3Ji4Yv6a6afuVB7DTNGl3+5W562lKiR2SWPev0tJmka2MqSwe82LCAwtvClzSVGYT4mUKVIdKqTmAV7mngY4atezKow3x58U8s1wvIx07QXVFxebXVKeWUNfLbz3wo2goecAoMlukujXjM8cRAYG4Ww++/yTkvsw90iL6u1w6u8BlESjAclMsXemqm6EWH/BBMOyv08TeO0VCYXxr4RKwvpSIW95dDGTTZLEJHnMWBrhU2RVZAXNgzUy4xMGTIkWJ+Y7s+x796+9cIVVPuezisWywwFYa1on4ZsHbITKPcQiovMQVtMfbJqYBqq0pShDCnOywLgTaYCKToYB+8+/AzNuI7TW0eTxfvF+cWfvj1/Q+vcZNYYe2MLa5ewnnF3pmzfeMr30GRkTCW7aux46bqzSQqjVAwT2D3OV2JRckseGSsE4qPFQvtSx6DI7CloesVbJRPU0kLGhUBDZRSDLYRUikazGI1Vio2oKiV43lkZUo1IRrSCMlh2VeXmnNGJuXLpp7I8mvaIthI0FnM17G5vGNxa2rAqaKBxFt5MZp5/hVQ4YLnEnCim7Lcu/4ciH374+P5vk7dfv/nr5OP5jz+ff3/x/uvvthQDZDuBaHydS6aryFdb/fCxKmjVK4Fk1iORG4pVkL0o0bTeGcHV2Jlqpa4XNHmyFMnNsVHtc63TRIzA8XGdn8j1y7tcqLC93Au+s4GkYOEN6b6tJW14gUxbZBbiU24XobdNZE/XKAN7tG08mCHNK5m8E7bu+bEizRvx3rcco6+K8javdl7QfmG4I7OQ/P8tvwFK0iRHloBDGTJtFCBKZS4NUfcDyxGPAZmcRH314/sP5/w+SNPq+48Xb2CkS0Cb+YIFT9OcDSv7uYhFElIe9bDCYsApEXPQyP0yXK2qcq8rYVPJK0KfuFJORobxpLVWL+iNXfiQVJbzBvUbo0BpLiSFkBW02f65uUqRyiKkT65WNkNFg936ZiSCM6aBCWRlII80EL1iMMYV/pZNc2ETHqOHWkDIiEONclL5Vk0GLGqbKtfXQClMja9BYjH+xL32QjOAxzADGNI0MDMhgUvLcv8RI4WYh1S0ZCYyzCY8l4MGDcYLx/LExl1t6ayViXtb/64s5M9aSvcDeXXo+e8RvSYa+GFE0x74AzPrY3yvBK88aLjtvXhmGQkRJ/dOPw7c7uxeIS/h7ql6WnZ6r9t7xAAeT7w06ef71BPtOa5sLAxL/BvSfMt+WDXY1Jo0TzFXBu14Mmxn1DnpDsbQ57PVJluIGivXWkrLqGogUG/bIkaSEHLYVdZO1YbYYi8aBE+1DWHfVem+d/ZxU9tMvonqrj1KR5cheqSwjEfufp3ONwQ6fKAnYzJ6SMHw/Ymnv9kWI4wWoaGfNiSBlUE3lCX8cbswDj4Eq2RoketvAA6Dd3AykbW3OWAMj+6aQclpvEIFjNvbrPKhxQnxe2uE/nb50N9fGjP2eFq8+aPLEuJapcGAr/spEUS2UqeCG+yttIXbojbHj0Pyak07x8ZCaJxZje821nyxv16JA1dHSZCwZSSDHtSvaga3H78wVn7liP1+Dshe1TwIOn5xwrbWNNlMGOOFAsDuYXychvYYeW1srdPbW4mCAJ1Z6k1NRU5gKat23RdmgAjOpwXYOkuBHjPOzmGEMnSw9m4VUJ79wy6h8n1dFh2SU2NCKbSpst+hAw2inYEZZXIOSg4LPLAMyXFDj96UHBJuPEUWGoS6mEHC2AlYh707hzY3GPoOcF6x1KRZZ3+vJfzfODM903c0kZKU0DGRuyP1AW6MjpOAiJQswBUFePrkQmUP9Gr8xM9xmBF7Ad4ZtteTW6L2coosIoP14aUcyKJ6gLvslhXul9GdYr6zx7pJU0aPMdhpvIZj+ZBSrUPR1UVZdMhbchWKZzcPi5Qp2QN3mx5pj0lc3mOo38zYmVWniXalCVn8njFu2eepGxRkBQTbm/PMGt15LmHOY/XNz2/enV+o87/96fz8DTxHdRfcW82bM2G6FIywj53KxqhJGIgBTQexsFS2DP3CtqYAS9XwqqATfKoGG7vnivhMMtncK88dZW8ZJNKb7gDqau0tA+DLme25bHeD3CyfxMvKSOqlSIc0FoMdVfG82UTQbc4szTwaOZ3f0ZOm1+ROJez9sMjKVWuwRY4zDjm0BsOPdnaXflCDvjiM7Q5X3vrS91QyqCLSBPjUEGWqAOnninJwERYu0V9Wok9UDWf2qjVHW82yj5BIguk8GxM8RZmwtkbLqFmCp4uO+D27GHYNnJYRuxzSlkmyUG1/Efgclj6GVPbeahbZOybjvgirtSVynqk7oaZJfqlhO4PmCbqgw86g+SYBzmDp405Ssa2HM+rBQN3Bt4MucYNqSikt1YQS4GgVZ/7B1sL04bfwOTxXVNvrqonMdfyZGyJUPhroIjJaCa/S1xGGS0e7YF4VB1UBAzpIuOUZ0RrhZlfgRKZLMeU2SQy1oObLR9SJl5UacpihmkGe35j8cRED6ICRiHebj7AFEdBXAtRhNfFhSZEZOUEqhiZvDzf8KHZhvuSHwjTzXLktD8epRiOMwFqPjuQybTr12BkzCmLo/5TJasAwqvnJ+dLR6yWNr036NMM9I5MEXaZfT0zetV58XygNQJAm56A0pdhWAuECl9AWTSUASK/DFY6ZygaPSCpbNJ7e2UEuMiVWAe/3KK+1FTVr6JqTrrA7QiyBCpFPe513uYLBI9LGFoM5JuzJqi1zyvctqKLEvUm6uJNO5OzivZIOweLiA51JdO7bSqc23FF1zC7Uy/by07wS++w69HPshfPV+rTDO/Fi+iyaBFBybXrZ8TNPyTNOdNJGWQTRa0FenL01wS0zgVxnWFMxpcQV/bkoFjB/JzINVrFUdRONSAOT/YjWF7d8dDOOooBNEZdtZ0zOM918azds8dm2iSn76GZDeKjDvC4Zw5NNBuxZL2nHpQKlMuJJmOhkYOAk7EIMKqelgxvYUOxFhFVjTCZvRnF9ne3DZnBbRI0cDWXzcViT2YO4U6xwXDqF7XetcA55WdoRW2TgtPRx35IZKGWO0nSZMc/UiXo1lIE+Q10eI7kM0rfFXs4DBo6QXibRqM5rdVcjoWm/gKU7VP1dpmH0SAlGj23dsyNknEiAGDpIKxImY5fmI2WbYSExQfqSPKY2maD8Y5oYYf7p/26xg4XMfdY0pQnFNjtVGYXjKmKHltiJKFr8EmVfFzTWoS2hwaB/qFIH6ppI67YqX03N0Km+1f0ioNZ6oJN9cyaMuuN/oGnugtW9+uaIY9l4IuYe9Z89O+4O3B6wsXfhNyIk8FDuRPrvaa6tykCwv5GqEzqg1vjGzpwmjbWGOR9O5skGw33rYY53LXABGytANG+DVnQV+qHXydYhb4DfWWDW1Esok2EiMSuN6RR4taPB2NK2r3ROAJq44BaFriBxr1WvavG/hWPxfZy/pVwdxsAq9j7FHQxETZsQxXoIWRko5nWb/1gEb8aD8uwRv6Wpw0uamuq6pI/CP3wkMbCMGWYNxA7dCn5YYi3hqzGyUOOTOJlMRvFEX8qlJt/KZ3zVlsGgyTKQvtjLHuh9EDXEq52Q/0tvy0davtqt4C/m4V6M49VKJ/qQrNNWQNHtgj9iU+Sft6KY/Iodxflfw8qCnoaUGcQHugLk1AQpnRzR4jmoLmboEA1cwA+iVd3oCEEyqgXrS3x3mZmS5JIiUlkEWQXiuJuG920O7SaQnD0+IUfKUH1ZLvtHmpeTKXVQYc6OoSkm2pTMwhGbF+G41ILg2rhSb/qIyeAJ3dftpHLgycwOqif1qWHPbNAQASSu7Ib9ACpTq4VhTN5fzSZOxBo2luxUm8AlNie2rX522vvSI7bARalVLCyHG62jiLW1NW5KACl0UHNSBsZmNDk1Tgh/Z12Av1ttyYJ8TEO8gve0US5fUoF7W6MNZ7QNfglLyhacxjEv647anoaFkwzpZ/m49UBJcWwgnSYZRBzzjuqJ2dBUPTQwyU6oTb9+NdxHY2NIyYIlVmS3NrA2uiHRF5NzCSOhNxZqcYe/vCIEC7FilLJtEerPle3UwDUG8ZowNtaU0iIRP1tjG7MhKBiU2RY4XKdFxuQm09iJynqg9LPZBEUUqzksLtysBMb512lspq7itrIMquGqZiq+MMm5yqTD31JGTZnSLAZPSNZ1DgMckSkgY0hXZhCxMKwFHKAcfXEZG/sSk9UsfZK53v6sKhaYxq/VNo6OYAs3lhKrYlTu6GClv5kG5GBzfkOlmcYGGmw/Dc3eIcH4vjn7R3OH+UkMKvPm7yr93T+wZbHCN1rH38q72qWp2BBB0lHdhKyRe2SgqbvdNFDnHtiqbwK5NbO0NE+RS53j8MRdZXfv6L1dV3QeGBVPvbFkdif9C5ObDsd0oHPWCcElPhZHOhiDp4QPCXNIOhf9cSOdty5Sy5g1p5XwBNCBi93esniWE2RAbiA3c4WTIZcbQCDIdWOLNoNFtAF6gVWYYExcCAvFHELJpH9DjUVJ5dOgQaGwLWZU65LU624GDsMoHImgo6H855pYChqyH4lNhOqhyjOCITKK59s94i8/YI8QXYQ0VXyRctL9M9Vze6dUraoZO6QwO025kRRLp95e70Djn9XrM6s8anBHt0a0TKIZ0domcIcb1h+3NgG2p7BDGSAUZlW7ipPp0+bZ1NkirTSRySyDF1Sim7RdfTnUf0XgxmltKwvtPozN+rvb8kGokvgVExNWE/g7v68TPmN3nI2mbEp78W4mrVmHjPBndlNuFopgHUBI8VUHEsiCHYU15pQ2YDly3qey6jYtuzmS/gWyk7/TR2OKlyQpPXPkGN+gY/D3Bk15gpylznSzJkeL8ccak50KBLDUti/98TbvKvhlQiErFMsgFw/HLMrzC9Z+c5LwziFD0KG/C0CzB56NQMT+BqQbGicmcAq2RcS6UV1qauPZ/mOMx/0H6NPHKob4d38h88crhiWoq90IoJT1zvc3Irb5zkBMQ0A8yzYH+4fECfla1dCpBnIsi2Mg9T/h9gCT4AjxQtlhXtlu5cCFmb22esRUVpvIY45kPCxI1UMsFp9YQBBpyEiJ06oud97SJo3pjrKzlfBkESzd0RmFUSnt7O6vS9hMV5JVqV23q62kWPkLAACta9RFE9JX3Khep1sJVQ9zoXJmh5hgXe+cjuDjPD7Oy9hSFrWFWRHi7/sAMJ//6sKdpkjbHFWEnsZ2p8fwVHP46Vgr+W04qwXmTibEpsmEYd/JhCO3Ew39Sn7T4Vj64Tpch+twHa7DdbgO1+E6XIfrcB2uw3W4DtfhOlyH63AdrsN1uA7X4Tpc+67/BqWysTwAeAAA"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_file(filename, path_hint, require_hint=True):
    matches = []
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            full = os.path.join(root, filename)
            if path_hint.lower() in full.lower():
                print(f"[FOUND] {filename} ({path_hint}): {full}", flush=True)
                return full
            matches.append(full)
    if not require_hint and matches:
        print(f"[FOUND] {filename} (fallback): {matches[0]}", flush=True)
        return matches[0]
    raise FileNotFoundError(f"Missing {filename} (hint={path_hint}, candidates={matches})")


def find_qwen_base_dir(expected_hidden_size, model_type_substr="qwen"):
    for root, _, files in os.walk("/kaggle/input"):
        if "config.json" in files and ("model.safetensors" in files or "model.safetensors.index.json" in files):
            if "embedding" in root.lower():
                continue
            try:
                with open(os.path.join(root, "config.json"), "r") as f:
                    cfg = json.load(f)
                if cfg.get("hidden_size") == expected_hidden_size and model_type_substr.lower() in str(cfg.get("model_type", "")).lower():
                    print(f"[FOUND] {model_type_substr} Base (hidden={expected_hidden_size}): {root}", flush=True)
                    return root
            except Exception:
                pass
    raise FileNotFoundError(f"Missing {model_type_substr} base (hidden={expected_hidden_size})")


def find_qwen_emb_dir():
    for root, _, files in os.walk("/kaggle/input"):
        if "config.json" in files and "embedding" in root.lower() and ("model.safetensors" in files or "model.safetensors.index.json" in files):
            try:
                with open(os.path.join(root, "config.json"), "r") as f:
                    cfg = json.load(f)
                if cfg.get("hidden_size") == 4096 and "qwen" in str(cfg.get("model_type", "")).lower():
                    print(f"[FOUND] Qwen3-Embedding-8B: {root}", flush=True)
                    return root
            except Exception:
                pass
    return None


def find_qwen_lora_dir(expected_hidden_size, path_hint, required=True):
    for root, _, files in os.walk("/kaggle/input"):
        if "classes.json" in files and "fold_0" in os.listdir(root):
            try:
                with open(os.path.join(root, "classes.json"), "r") as f:
                    meta = json.load(f)
                if int(meta.get("hidden_size", -1)) == expected_hidden_size and path_hint.lower() in root.lower():
                    print(f"[FOUND] Qwen3 LoRA ({path_hint}): {root}", flush=True)
                    return root
            except Exception:
                pass
    if not required:
        print(f"[INFO] Optional Qwen3 LoRA ({path_hint}, hidden={expected_hidden_size}) not attached; skipping.", flush=True)
        return None
    raise FileNotFoundError(f"Missing Qwen3 LoRA ({path_hint}, hidden={expected_hidden_size})")


def find_encoder_dir(expected_hidden_size, expected_model_type, path_hint):
    for root, _, files in os.walk("/kaggle/input"):
        if "fold_0.pt" in files and "config.json" in files and path_hint.lower() in root.lower():
            try:
                with open(os.path.join(root, "config.json"), "r") as f:
                    cfg = json.load(f)
                if cfg.get("hidden_size") == expected_hidden_size and expected_model_type in str(cfg.get("model_type", "")).lower():
                    print(f"[FOUND] {expected_model_type} ({path_hint}, hidden={expected_hidden_size}): {root}", flush=True)
                    return root
            except Exception:
                pass
    raise FileNotFoundError(f"Missing {expected_model_type} ({path_hint}, hidden={expected_hidden_size})")


class PreTokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}


def make_sorted_loader(test_texts, tokenizer, max_len, batch_size):
    enc = tokenizer([str(t) for t in test_texts], truncation=True, max_length=max_len, padding=False)
    lengths = np.asarray([len(x) for x in enc["input_ids"]], dtype=np.int32)
    order = np.argsort(lengths, kind="mergesort")
    inv_order = np.argsort(order, kind="mergesort")
    loader = DataLoader(
        PreTokenizedDataset([enc["input_ids"][i] for i in order], [enc["attention_mask"][i] for i in order]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    return loader, inv_order


def make_classifier_head(hidden_size, n_classes):
    return nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Dropout(0.1), nn.Linear(hidden_size * 2, n_classes))


def resolve_module_name(lora_prefix, name_to_mod):
    cands = [lora_prefix]
    for strip_prefix in ["base_model.model.model.", "base_model.model.", "base_model.", "model."]:
        if lora_prefix.startswith(strip_prefix):
            cands.append(lora_prefix[len(strip_prefix):])
    for c in cands:
        if c in name_to_mod:
            return c
    for mname in name_to_mod:
        if lora_prefix.endswith("." + mname) or lora_prefix.endswith(mname):
            return mname
    return None


def predict_qwen3_ensemble(test_texts, classes, qwen_base_dir, qwen_lora_dir, device, amp_dtype, use_amp, batch_size=24, tag="QWEN3"):
    t0 = time.time()
    with open(os.path.join(qwen_lora_dir, "classes.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, f"{tag} class mismatch!"
    hidden_size = int(meta["hidden_size"])
    max_len = int(meta.get("max_len", MAX_LEN))

    tokenizer = AutoTokenizer.from_pretrained(qwen_lora_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    test_loader, inv_order = make_sorted_loader(test_texts, tokenizer, max_len, batch_size)

    config = AutoConfig.from_pretrained(qwen_base_dir)
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = False
    keep_layers = int(meta.get("keep_layers", config.num_hidden_layers))
    if keep_layers < config.num_hidden_layers:
        print(f"[{tag}] Truncating base layers: {config.num_hidden_layers} -> {keep_layers}", flush=True)
        config.num_hidden_layers = keep_layers
    model_dtype = amp_dtype if use_amp else torch.float32

    base_model = AutoModel.from_pretrained(qwen_base_dir, config=config, torch_dtype=model_dtype).to(device)
    base_model.eval()
    name_to_mod = dict(base_model.named_modules())

    with open(os.path.join(qwen_lora_dir, "fold_0", "adapter_config.json"), "r") as f:
        acfg = json.load(f)
    scaling = float(acfg["lora_alpha"]) / float(acfg["r"])

    sample_lora = load_safetensors(os.path.join(qwen_lora_dir, "fold_0", "adapter_model.safetensors"))
    target_map = {}
    for k in sample_lora:
        if k.endswith(".lora_A.weight"):
            prefix = k[:-len(".lora_A.weight")]
            mname = resolve_module_name(prefix, name_to_mod)
            assert mname is not None, f"Unresolved LoRA key: {k}"
            target_map[prefix] = mname
    del sample_lora

    orig_weights = {mname: name_to_mod[mname].weight.data.cpu().clone() for mname in target_map.values()}
    head = make_classifier_head(hidden_size, len(classes)).to(device)
    head.eval()

    probs_qwen3 = np.zeros((len(test_texts), len(classes)), dtype=np.float64)
    for fold in range(5):
        fold_dir = os.path.join(qwen_lora_dir, f"fold_{fold}")
        lora_sd = load_safetensors(os.path.join(fold_dir, "adapter_model.safetensors"))
        for prefix, mname in target_map.items():
            lora_A = lora_sd[prefix + ".lora_A.weight"].to(device, dtype=torch.float32)
            lora_B = lora_sd[prefix + ".lora_B.weight"].to(device, dtype=torch.float32)
            w_orig = orig_weights[mname].to(device, dtype=torch.float32)
            name_to_mod[mname].weight.data.copy_((w_orig + scaling * (lora_B @ lora_A)).to(model_dtype))
        del lora_sd

        head.load_state_dict(torch.load(os.path.join(fold_dir, "classifier_head.pt"), map_location=device))
        fold_probs = []
        with torch.inference_mode():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    hidden = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
                    bsz = hidden.size(0)
                    last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
                    last_tok = hidden[torch.arange(bsz, device=device), last_idx].float()
                    mask_f = attention_mask.unsqueeze(-1).float()
                    mean_tok = (hidden.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
                    logits = head(torch.cat([last_tok, mean_tok], dim=-1))
                fold_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs_qwen3 += np.concatenate(fold_probs, axis=0)[inv_order] / 5.0
        print(f"  - {tag} Fold {fold + 1}/5 done.", flush=True)

    del base_model, orig_weights, head, test_loader, name_to_mod, target_map
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert np.isfinite(probs_qwen3).all(), f"Non-finite outputs from {tag}"
    print(f"[{tag}] Complete in {time.time() - t0:.1f}s", flush=True)
    return probs_qwen3


def predict_qwen3_multilayer_ensemble(
    test_texts,
    classes,
    qwen_base_dir,
    qwen_lora_dir,
    amp_dtype,
    use_amp,
    batch_size=16,
    tag="QWEN3-8B",
    extract_base_feats=False,
    extract_stage1_lora=None,
):
    t0 = time.time()
    with open(os.path.join(qwen_lora_dir, "classes.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, f"{tag} class mismatch!"
    hidden_size = int(meta["hidden_size"])
    max_len = int(meta.get("max_len", MAX_LEN))
    layer_indices = list(meta["layer_indices"])

    tokenizer = AutoTokenizer.from_pretrained(qwen_lora_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    test_loader, inv_order = make_sorted_loader(test_texts, tokenizer, max_len, batch_size)

    config = AutoConfig.from_pretrained(qwen_base_dir)
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = False
    keep_layers = int(meta.get("keep_layers", config.num_hidden_layers))
    if keep_layers < config.num_hidden_layers:
        print(f"[{tag}] Truncating base layers: {config.num_hidden_layers} -> {keep_layers}", flush=True)
        config.num_hidden_layers = keep_layers
    model_dtype = amp_dtype if use_amp else torch.float32

    use_auto_map = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_auto_map:
        base_model = AutoModel.from_pretrained(qwen_base_dir, config=config, torch_dtype=model_dtype, device_map="auto")
    else:
        single_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        base_model = AutoModel.from_pretrained(qwen_base_dir, config=config, torch_dtype=model_dtype).to(single_dev)
    base_model.eval()
    name_to_mod = dict(base_model.named_modules())
    embed_dev = base_model.get_input_embeddings().weight.device
    head_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(qwen_lora_dir, "fold_0", "adapter_config.json"), "r") as f:
        acfg = json.load(f)
    scaling = float(acfg["lora_alpha"]) / float(acfg["r"])

    sample_lora = load_safetensors(os.path.join(qwen_lora_dir, "fold_0", "adapter_model.safetensors"))
    target_map = {}
    for k in sample_lora:
        if k.endswith(".lora_A.weight"):
            prefix = k[:-len(".lora_A.weight")]
            mname = resolve_module_name(prefix, name_to_mod)
            assert mname is not None, f"Unresolved LoRA key: {k}"
            target_map[prefix] = mname
    del sample_lora

    class MultiLayerAuthorshipHead(nn.Module):
        def __init__(self, h_dim, num_cls, n_lyr):
            super().__init__()
            self.n_lyr = n_lyr
            self.layer_logits_last = nn.Parameter(torch.zeros(n_lyr))
            self.layer_logits_mean = nn.Parameter(torch.zeros(n_lyr))
            self.classifier = nn.Sequential(nn.LayerNorm(h_dim * 2), nn.Dropout(0.15), nn.Linear(h_dim * 2, num_cls))

        def forward(self, last_stack, mean_stack):
            w_last = torch.softmax(self.layer_logits_last, dim=0).view(1, self.n_lyr, 1)
            w_mean = torch.softmax(self.layer_logits_mean, dim=0).view(1, self.n_lyr, 1)
            return self.classifier(torch.cat([(last_stack * w_last).sum(dim=1), (mean_stack * w_mean).sum(dim=1)], dim=-1))

    head = MultiLayerAuthorshipHead(hidden_size, len(classes), len(layer_indices)).to(head_dev)
    head.eval()

    probs_qwen3 = np.zeros((len(test_texts), len(classes)), dtype=np.float64)
    prev_deltas = None
    extra_feats = None

    if extract_base_feats or extract_stage1_lora is not None:
        if extract_stage1_lora is not None:
            s1_sd = torch.load(extract_stage1_lora, map_location="cpu", weights_only=True)
            prev_deltas = {}
            for prefix, mname in target_map.items():
                mod = name_to_mod[mname]
                dev = mod.weight.device
                ka = prefix + ".lora_A.default.weight" if (prefix + ".lora_A.default.weight") in s1_sd else prefix + ".lora_A.weight"
                kb = prefix + ".lora_B.default.weight" if (prefix + ".lora_B.default.weight") in s1_sd else prefix + ".lora_B.weight"
                lora_A = s1_sd[ka].to(dev, dtype=torch.float32)
                lora_B = s1_sd[kb].to(dev, dtype=torch.float32)
                delta = (scaling * (lora_B @ lora_A)).to(mod.weight.dtype)
                mod.weight.data.add_(delta)
                prev_deltas[mname] = delta.detach().cpu()
            del s1_sd

        feat_chunks = []
        with torch.inference_mode():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(embed_dev)
                attention_mask = batch["attention_mask"].to(embed_dev)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                mean_list = []
                for lyr in layer_indices:
                    h = hs[lyr]
                    dev_l = h.device
                    am_l = attention_mask if dev_l == embed_dev else attention_mask.to(dev_l)
                    mask_l = am_l.unsqueeze(-1).float()
                    den_l = mask_l.sum(dim=1).clamp(min=1e-6)
                    mean_l = F.layer_norm((h.float() * mask_l).sum(dim=1) / den_l, (hidden_size,)).cpu()
                    mean_list.append(mean_l)
                feat_chunks.append(torch.cat(mean_list, dim=-1).numpy().astype(np.float16))
        extra_feats = np.concatenate(feat_chunks, axis=0)[inv_order]
        assert np.isfinite(extra_feats).all(), f"Non-finite extracted features from {tag}"
        print(f"  - {tag} Gauss feature pass done shape={extra_feats.shape}", flush=True)

    for fold in range(5):
        fold_dir = os.path.join(qwen_lora_dir, f"fold_{fold}")
        if prev_deltas is not None:
            for mname, d_cpu in prev_deltas.items():
                mod = name_to_mod[mname]
                mod.weight.data.sub_(d_cpu.to(mod.weight.device, dtype=mod.weight.dtype))
            del prev_deltas

        lora_sd = load_safetensors(os.path.join(fold_dir, "adapter_model.safetensors"))
        prev_deltas = {}
        for prefix, mname in target_map.items():
            mod = name_to_mod[mname]
            dev = mod.weight.device
            lora_A = lora_sd[prefix + ".lora_A.weight"].to(dev, dtype=torch.float32)
            lora_B = lora_sd[prefix + ".lora_B.weight"].to(dev, dtype=torch.float32)
            delta = (scaling * (lora_B @ lora_A)).to(mod.weight.dtype)
            mod.weight.data.add_(delta)
            prev_deltas[mname] = delta.detach().cpu()
        del lora_sd

        head.load_state_dict(torch.load(os.path.join(fold_dir, "classifier_head.pt"), map_location=head_dev))
        fold_probs = []
        with torch.inference_mode():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(embed_dev)
                attention_mask = batch["attention_mask"].to(embed_dev)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                bsz = input_ids.size(0)
                last_list, mean_list = [], []
                for lyr in layer_indices:
                    h = hs[lyr]
                    dev_l = h.device
                    am_l = attention_mask if dev_l == embed_dev else attention_mask.to(dev_l)
                    idx_l = torch.arange(bsz, device=dev_l)
                    last_idx_l = (am_l.sum(dim=1) - 1).clamp(min=0)
                    mask_l = am_l.unsqueeze(-1).float()
                    den_l = mask_l.sum(dim=1).clamp(min=1e-6)
                    last_l = F.layer_norm(h[idx_l, last_idx_l].float(), (hidden_size,)).to(head_dev)
                    mean_l = F.layer_norm((h.float() * mask_l).sum(dim=1) / den_l, (hidden_size,)).to(head_dev)
                    last_list.append(last_l)
                    mean_list.append(mean_l)
                logits = head(torch.stack(last_list, dim=1), torch.stack(mean_list, dim=1))
                fold_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs_qwen3 += np.concatenate(fold_probs, axis=0)[inv_order] / 5.0
        print(f"  - {tag} Fold {fold + 1}/5 done.", flush=True)

    del prev_deltas, base_model, head, test_loader, name_to_mod, target_map
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert np.isfinite(probs_qwen3).all(), f"Non-finite outputs from {tag}"
    print(f"[{tag}] Complete in {time.time() - t0:.1f}s", flush=True)
    if extract_base_feats or extract_stage1_lora is not None:
        return probs_qwen3, extra_feats
    return probs_qwen3


def predict_modernbert_ensemble(test_texts, classes, bert_dir, device, amp_dtype, use_amp, batch_size=48, tag="BERT"):
    t0 = time.time()
    with open(os.path.join(bert_dir, "classes.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, f"{tag} class mismatch!"

    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    config = AutoConfig.from_pretrained(bert_dir)
    test_loader, inv_order = make_sorted_loader(test_texts, tokenizer, MAX_LEN, batch_size)

    model = AutoModelForSequenceClassification.from_config(config).to(device)
    model.eval()
    probs_bert = np.zeros((len(test_texts), len(classes)), dtype=np.float64)
    for fold in range(5):
        state = {k: v.float() for k, v in torch.load(os.path.join(bert_dir, f"fold_{fold}.pt"), map_location="cpu").items()}
        model.load_state_dict(state)
        del state
        fold_probs = []
        with torch.inference_mode():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                fold_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs_bert += np.concatenate(fold_probs, axis=0)[inv_order] / 5.0
        print(f"  - {tag} Fold {fold + 1}/5 done.", flush=True)

    del model, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert np.isfinite(probs_bert).all(), f"Non-finite outputs from {tag}"
    print(f"[{tag}] Complete in {time.time() - t0:.1f}s", flush=True)
    return probs_bert


def predict_deberta_multilayer_ensemble(test_texts, classes, deb_dir, device, batch_size=32, tag="DeBERTa-v3-large"):
    t0 = time.time()
    with open(os.path.join(deb_dir, "classes.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, f"{tag} class mismatch!"
    layers = list(meta["layer_indices"])
    h_dim = int(meta["hidden_size"])
    max_len = int(meta.get("max_len", MAX_LEN))

    tokenizer = AutoTokenizer.from_pretrained(deb_dir)
    config = AutoConfig.from_pretrained(deb_dir)
    test_loader, inv_order = make_sorted_loader(test_texts, tokenizer, max_len, batch_size)

    class DebertaMultiLayerClassifier(nn.Module):
        def __init__(self, backbone_model):
            super().__init__()
            self.backbone = backbone_model
            self.layers = layers
            self.h_dim = h_dim
            self.head = nn.Sequential(
                nn.LayerNorm(self.h_dim * len(self.layers)),
                nn.Dropout(0.15),
                nn.Linear(self.h_dim * len(self.layers), len(classes)),
            )

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hs = out.hidden_states
            mask_f = attention_mask.unsqueeze(-1).float()
            den = mask_f.sum(dim=1).clamp(min=1e-6)
            vecs = [F.layer_norm((hs[lyr].float() * mask_f).sum(dim=1) / den, (self.h_dim,)) for lyr in self.layers]
            return self.head(torch.cat(vecs, dim=-1))

    model = DebertaMultiLayerClassifier(AutoModel.from_config(config)).to(device)
    model.eval()

    probs_deb = np.zeros((len(test_texts), len(classes)), dtype=np.float64)
    for fold in range(5):
        state = {k: v.float() for k, v in torch.load(os.path.join(deb_dir, f"fold_{fold}.pt"), map_location="cpu").items()}
        model.load_state_dict(state)
        del state
        fold_probs = []
        with torch.inference_mode():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                fold_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs_deb += np.concatenate(fold_probs, axis=0)[inv_order] / 5.0
        print(f"  - {tag} Fold {fold + 1}/5 done.", flush=True)

    del model, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert np.isfinite(probs_deb).all(), f"Non-finite outputs from {tag}"
    print(f"[{tag}] Complete in {time.time() - t0:.1f}s", flush=True)
    return probs_deb


def shape_view(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_view(text):
    return " ".join(
        t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
        for t in re.findall(r"\w+|[^\w\s]", text)
    )


def prepare_texts(texts, view):
    if view == "shape":
        return [shape_view(t) for t in texts]
    if view == "function":
        return [function_view(t) for t in texts]
    return texts


def sparse_group_centre(X, groups, lo=2, hi=18):
    X = csr_matrix(X)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep):
        return X.copy()
    col = np.full(len(cnt), -1)
    col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)))
    means = diags(1.0 / cnt[keep]) @ (M.T @ X)
    return (X - M @ means).tocsr()


def style_groups(texts, thresh=0.90):
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    try:
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=100000, dtype=np.float32)
        X = vec.fit_transform(texts)
        D = np.clip(1.0 - (X @ X.T).toarray(), 0.0, 1.0)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        D = np.ones((len(texts), len(texts)), dtype=np.float32) - np.eye(len(texts), dtype=np.float32)
    med = max(float(np.median(D[np.triu_indices(len(texts), 1)])), 1e-6)
    D = np.asarray(D / med, dtype=np.float64).copy()
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), thresh, criterion="distance")


def capped_agglo(S, cap=18, thr=0.08):
    n = len(S)
    A = S.astype(np.float32).copy()
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32)
    alive = np.ones(n, bool)
    lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new
        A[:, i] = new
        A[i, i] = -np.inf
        A[j] = -np.inf
        A[:, j] = -np.inf
        size[i] += size[j]
        alive[j] = False
        lab[lab == j] = i
    return lab


def cluster_sinkhorn(logp, groups_c, n_iters=3, alpha=0.65, lo=6, hi=18):
    out = logp.copy()
    _, inv, cnt = np.unique(groups_c, return_inverse=True, return_counts=True)
    for g in np.flatnonzero((cnt >= lo) & (cnt <= hi)):
        idx = np.flatnonzero(inv == g)
        z = out[idx].copy()
        for _ in range(n_iters):
            z = log_softmax(z, axis=1)
            z = z - alpha * z.mean(axis=0, keepdims=True)
        out[idx] = log_softmax(z, axis=1)
    return out


def global_sinkhorn(logp, mix=0.35):
    P = np.exp(logp - logp.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    Q = P.copy()
    target = np.full(P.shape[1], 1.0 / P.shape[1])
    for _ in range(100):
        Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return log_softmax((1 - mix) * np.log(P + 1e-12) + mix * np.log(Q + 1e-12), axis=1)


def predict_stylegram_v1_with_clusters(texts, classes, stylegram_dir, G_agglo):
    t0 = time.time()
    with open(os.path.join(stylegram_dir, "bundle_meta.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, "StyleGram v1 class mismatch!"
    prepared = {v: prepare_texts(texts, v) for v in ["char", "word", "shape", "function"]}
    sg_lp = np.zeros((len(texts), len(classes)), dtype=np.float64)
    n_folds = int(meta["n_folds"])
    w_lr = float(meta["lr_weight"])
    w_svc = float(meta["svc_weight"])
    for fold in range(n_folds):
        f_dir = os.path.join(stylegram_dir, f"fold_{fold}")
        vecs = joblib.load(os.path.join(f_dir, "vectorizers.joblib"))
        X_parts = [item["vec"].transform(prepared[item["view"]]) * float(item["weight"]) for item in vecs]
        X = hstack(X_parts, format="csr")
        X_c = sparse_group_centre(X, G_agglo, lo=2, hi=18)
        mods = np.load(os.path.join(f_dir, "models.npz"))
        z_lr = log_softmax(X_c @ mods["lr_coef"].T + mods["lr_intercept"], axis=1)
        z_svc = log_softmax(X_c @ mods["svc_coef"].T + mods["svc_intercept"], axis=1)
        sg_lp += log_softmax(w_lr * z_lr + w_svc * z_svc, axis=1) / n_folds
    print(f"[StyleGram v1] 5-fold inference complete in {time.time() - t0:.1f}s", flush=True)
    return log_softmax(sg_lp, axis=1)


def predict_stylegram_v2_and_clusters(texts, classes, stylegram_dir):
    t0 = time.time()
    with open(os.path.join(stylegram_dir, "bundle_meta.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, "StyleGram v2 class mismatch!"
    families = {fam: np.asarray(idxs, dtype=int) for fam, idxs in meta["families"].items()}

    if len(texts) >= 40:
        try:
            vc = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
            Xc = vc.fit_transform(texts)
            G_agglo = capped_agglo((Xc @ Xc.T).toarray(), 18, 0.08)
        except ValueError:
            G_agglo = np.arange(len(texts))
    else:
        G_agglo = np.arange(len(texts))

    G_style = style_groups(texts, thresh=0.90)
    _, inv_s, cnt_s = np.unique(G_style, return_inverse=True, return_counts=True)
    in_clust = (cnt_s[inv_s] >= 2) & (cnt_s[inv_s] <= 18)

    _, cnt_a = np.unique(G_agglo, return_counts=True)
    print(
        f"[CLUSTERS] n={len(texts)}, agglo={len(cnt_a)} (max={cnt_a.max()}), "
        f"style_in_2_18={int(in_clust.sum())}/{len(texts)}",
        flush=True,
    )

    vecs = joblib.load(os.path.join(stylegram_dir, "vectorizers.joblib"))
    prepared = {v: prepare_texts(texts, v) for v in ["char", "word", "shape", "function"]}
    X_te = hstack([item["vec"].transform(prepared[item["view"]]) * float(item["weight"]) for item in vecs], format="csr")
    X_te_c = sparse_group_centre(X_te, G_style, lo=2, hi=18)

    n_folds = int(meta["n_folds"])
    sg_lp = np.zeros((len(texts), len(classes)), dtype=np.float64)
    gates_lp = {fam: np.zeros((len(texts), len(idxs)), dtype=np.float64) for fam, idxs in families.items()}

    for fold in range(n_folds):
        mods = np.load(os.path.join(stylegram_dir, f"fold_{fold}", "models.npz"))
        lp_p = log_softmax(X_te @ mods["svc_p_coef"].T + mods["svc_p_intercept"], axis=1)
        lp_c = log_softmax(X_te_c @ mods["svc_c_coef"].T + mods["svc_c_intercept"], axis=1)
        sg_lp += np.where(in_clust[:, None], lp_c, lp_p) / n_folds

        for fam, idxs in families.items():
            d = X_te @ mods[f"gate_{fam}_coef"].T + mods[f"gate_{fam}_intercept"]
            if len(idxs) == 2:
                d = d.ravel()
                d = np.column_stack([-d, d])
            d = d - d.mean(axis=1, keepdims=True)
            sc = np.sqrt(np.mean(d * d, axis=1, keepdims=True))
            gates_lp[fam] += log_softmax(d / np.maximum(sc, 0.25), axis=1) / n_folds

    print(f"[StyleGram v2] 5-fold inference complete in {time.time() - t0:.1f}s", flush=True)
    return log_softmax(sg_lp, axis=1), gates_lp, families, G_agglo


def apply_conditional_family_gates(logp, gates_lp, families, lam=0.35):
    if lam <= 0:
        return logp
    out = logp.copy()
    for fam, idxs in families.items():
        base = out[:, idxs]
        base_lp = log_softmax(base, axis=1)
        sp = np.sort(np.exp(base_lp), axis=1)
        conf_gate = np.clip((0.45 - (sp[:, -1] - sp[:, -2])) / 0.45, 0.0, 1.0)
        strength = min(0.45, lam / (1.0 + lam)) * conf_gate
        gate_fold_lp = log_softmax(gates_lp[fam], axis=1)
        mixed_lp = log_softmax((1.0 - strength[:, None]) * base_lp + strength[:, None] * gate_fold_lp, axis=1)
        out[:, idxs] = logsumexp(base, axis=1)[:, None] + mixed_lp
    return out


def setup_v1_assets():
    code_dir = Path("/kaggle/temp/v1-code")
    code_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(V1_CODE_B64)), mode="r:gz") as tf:
        tf.extractall(code_dir)

    cands = sorted(glob.glob("/kaggle/input/**/ntl-bin/ntl-segrun", recursive=True))
    ok = [
        p for p in cands
        if sha256_file(p) == NTL_SEGRUN_SHA256 and Path(p).with_name("lib").joinpath("libcublasLt.so.13").exists()
    ]
    assert ok, f"Valid ntl-segrun not found among {cands}"
    src_bin = Path(ok[0])
    bin_dir = Path("/kaggle/temp/v1-bin")
    bin_dir.mkdir(parents=True, exist_ok=True)
    dst_bin = bin_dir / "ntl-segrun"
    shutil.copy(src_bin, dst_bin)
    os.chmod(dst_bin, 0o755)
    v1_env = dict(os.environ, LD_LIBRARY_PATH=f"{src_bin.with_name('lib')}:" + os.environ.get("LD_LIBRARY_PATH", ""))

    def link_segs(pattern, expected, name):
        seen = {}
        for p in sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True)):
            seen.setdefault(os.path.basename(p), p)
        assert len(seen) == expected, f"{name}: found {len(seen)} != {expected}"
        d = Path(f"/kaggle/temp/v1-segs-{name}")
        d.mkdir(parents=True, exist_ok=True)
        for b, p in seen.items():
            if not (d / b).exists():
                (d / b).symlink_to(p)
        return d

    flash_dir = link_segs("flash_iq3xxs-seg*-of-06-*.gguf", 6, "flash")
    glm53_dir = link_segs("glm53_iq3xxs-seg*-of-19-*.gguf", 19, "glm53")
    bundle_dir = Path(find_file("bundle_meta.json", path_hint="exp010-glm-models")).parent
    return code_dir, bin_dir, v1_env, flash_dir, glm53_dir, bundle_dir


def predict_bundle_logp(bundle_dir, flash_csv, glm53_csv, sel_ids):
    with open(bundle_dir / "bundle_meta.json", "r") as f:
        meta = json.load(f)
    f_fl = pd.read_csv(flash_csv, dtype={"id": str}).set_index("id").loc[sel_ids]
    f_53 = pd.read_csv(glm53_csv, dtype={"id": str}).set_index("id").loc[sel_ids]
    df_feat = pd.concat([f_fl.add_prefix("flash:"), f_53.add_prefix("glm53:")], axis=1)
    for c in meta["diff_cols_ext"]:
        df_feat[f"d:{c}"] = df_feat[f"flash:{c}"] - df_feat[f"glm53:{c}"]
    df_feat["flash:drift"] = df_feat["flash:lp_mean_second"] - df_feat["flash:lp_mean_first"]
    df_feat["glm53:drift"] = df_feat["glm53:lp_mean_second"] - df_feat["glm53:lp_mean_first"]
    df_feat["d:drift"] = df_feat["flash:drift"] - df_feat["glm53:drift"]

    X = df_feat[meta["feature_cols"]].to_numpy(np.float64)
    assert np.isfinite(X).all(), "Non-finite GLM likelihood features!"

    n_folds = int(meta["n_folds"])
    n_cls = len(meta["classes"])
    lp_lr = np.zeros((len(X), n_cls), dtype=np.float64)
    lp_lgb = np.zeros((len(X), n_cls), dtype=np.float64)
    lp_cb = np.zeros((len(X), n_cls), dtype=np.float64)

    for fold in range(n_folds):
        lf = meta["lr_folds"][fold]
        xs = (X - np.asarray(lf["mean"])) / np.asarray(lf["scale"])
        z_lr = xs @ np.asarray(lf["coef"]).T + np.asarray(lf["intercept"])
        lp_lr += log_softmax(z_lr, axis=1) / n_folds

        bst = lgb.Booster(model_file=str(bundle_dir / f"lgb_fold_{fold}.txt"))
        p_lgb = np.asarray(bst.predict(X), dtype=np.float64)
        lp_lgb += np.log(np.clip(p_lgb, 1e-7, 1.0)) / n_folds

        cb = CatBoostClassifier()
        cb.load_model(str(bundle_dir / f"cb_fold_{fold}.cbm"))
        p_cb = np.asarray(cb.predict_proba(X), dtype=np.float64)
        lp_cb += np.log(np.clip(p_cb, 1e-7, 1.0)) / n_folds

    w = meta["weights"]
    ll_lp = log_softmax(w["lr"] * lp_lr + w["lgb"] * lp_lgb + w["cb"] * lp_cb, axis=1)
    return meta, ll_lp


def run_glm_stage(test_df, base_logp, classes, deadline_mono):
    code_dir, bin_dir, v1_env, flash_dir, glm53_dir, bundle_dir = setup_v1_assets()
    with open(bundle_dir / "bundle_meta.json", "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, "Class order mismatch between base & GLM bundle!"

    max_tok = int(meta["max_text_tokens"])
    tpc = float(meta["tok_per_char"])
    is_small_commit = len(test_df) <= 20

    ids = test_df["id"].astype(str).tolist()
    glm_idxs = [classes.index("glm-5.3"), classes.index("glm-5.3-flash")]
    probs_base = np.exp(base_logp)
    glm_mass = probs_base[:, glm_idxs].sum(1)
    order = np.argsort(-glm_mass, kind="stable")

    def time_left():
        return deadline_mono - time.monotonic() - RESERVE_S

    rows_i = np.minimum(test_df["text"].fillna("").astype(str).str.len().to_numpy() * tpc * 1.05, max_tok) + 13
    cum = np.cumsum(rows_i[order])
    fits = 1.1 * (FIXED_S_FLASH + FIXED_S_GLM53 + (ROW_S_FLASH + ROW_S_GLM53) * cum) <= time_left()

    if is_small_commit:
        n_sel = len(ids)
    else:
        n_sel = int(min(fits.sum(), max(int(round(MAX_COVER * len(ids))), 1)))
        if n_sel < MIN_TEXTS:
            print(f"[GLM FALLBACK] Only {n_sel} texts fit; keeping Part 2 submission.", flush=True)
            return None

    sel = np.sort(order[:n_sel])
    sel_ids = [ids[i] for i in sel]
    print(f"[GLM SELECT] {n_sel}/{len(ids)} texts ({n_sel/len(ids):.1%}), left={time_left():.0f}s", flush=True)

    work_dir = Path("/kaggle/temp/v1-work")
    out_dir = Path("/kaggle/working/v1")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    reserve_flash = 0.0 if is_small_commit else 1.05 * (FIXED_S_GLM53 + ROW_S_GLM53 * cum[min(MIN_TEXTS, n_sel) - 1])
    for tag, segs_glob, reserve in [("flash", str(flash_dir / "*.gguf"), reserve_flash), ("glm53", str(glm53_dir / "*.gguf"), 0.0)]:
        if tag == "glm53" and not is_small_commit:
            m = int(min(n_sel, (1.1 * (FIXED_S_GLM53 + ROW_S_GLM53 * cum[:n_sel]) <= time_left()).sum()))
            if m < MIN_TEXTS:
                print(f"[GLM FALLBACK] After Flash only {m} fit for GLM-5.3; keeping Part 2.", flush=True)
                return None
            if m < n_sel:
                n_sel = m
                sel = np.sort(order[:n_sel])
                sel_ids = [ids[i] for i in sel]

        budget = max(time_left() - reserve, 1.0)
        q_csv = work_dir / f"query_{tag}.csv"
        test_df.iloc[sel][["id", "text"]].to_csv(q_csv, index=False)
        cmd = [
            sys.executable, str(code_dir / "run_pipeline.py"),
            "--bin", str(bin_dir),
            "--segs", segs_glob,
            "--csv", str(q_csv),
            "--out", str(out_dir),
            "--tag", tag,
            "--work", str(work_dir / "state"),
            "--seq-batch", str(SEQ_BATCH),
            "--state-dtype", "auto",
            "--max-text-tokens", str(max_tok),
            "--prefix", PREFIX,
            "--budget-s", f"{budget:.0f}",
        ]
        t_p = time.monotonic()
        res = subprocess.run(cmd, env=v1_env)
        print(f"[GLM {tag.upper()}] exit={res.returncode} in {time.monotonic() - t_p:.0f}s", flush=True)
        if res.returncode == 3 and not is_small_commit:
            print(f"[GLM FALLBACK] {tag} hit budget limit; keeping Part 2.", flush=True)
            return None
        assert res.returncode == 0, f"GLM {tag} failed with code {res.returncode}"

    _, ll_lp = predict_bundle_logp(bundle_dir, out_dir / "flash_features.csv", out_dir / "glm53_features.csv", sel_ids)
    w_blend = float(meta["blend_weight"])
    before = base_logp.argmax(1)
    after = before.copy()
    after[sel] = (base_logp[sel] + w_blend * ll_lp).argmax(1)

    glm_arr = np.asarray(glm_idxs)
    glm_rel = np.isin(before, glm_arr) | np.isin(after, glm_arr)
    blocked = (before != after) & ~glm_rel
    after[blocked] = before[blocked]

    changed = float((before != after).mean())
    print(
        f"[GLM APPLIED] n={len(sel_ids)} changed={changed:.3f} blocked_non_glm={int(blocked.sum())} "
        f"glm53_cnt={int((after == glm_idxs[0]).sum())} flash_cnt={int((after == glm_idxs[1]).sum())}",
        flush=True,
    )
    if not is_small_commit:
        assert changed <= MAX_CHANGED, f"Sanity guard: changed={changed:.1%} > {MAX_CHANGED:.0%}"
    return after


def extract_qwen_emb8b(texts, emb_dir, amp_dtype, use_amp):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(emb_dir)
    auto, plain = tok("hello")["input_ids"], tok("hello", add_special_tokens=False)["input_ids"]
    eos = auto[-1] if (len(auto) > len(plain) and auto[-1] != plain[-1]) else (tok.eos_token_id if tok.eos_token_id is not None else 151643)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    cfg = AutoConfig.from_pretrained(emb_dir)
    cfg.use_cache = False
    model_dtype = amp_dtype if use_amp else torch.float32
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = AutoModel.from_pretrained(emb_dir, config=cfg, torch_dtype=model_dtype, device_map="auto").eval()
    else:
        single_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(emb_dir, config=cfg, torch_dtype=model_dtype).to(single_dev).eval()
    embed_dev = model.get_input_embeddings().weight.device
    hid = int(getattr(model.config, "hidden_size", 4096))

    enc = tok([str(t) for t in texts], truncation=True, max_length=511)["input_ids"]
    ids = [x if (x and x[-1] == eos) else (x + [eos]) for x in enc]
    order = np.argsort([len(x) for x in ids], kind="stable")
    emb = np.zeros((len(texts), hid), dtype=np.float16)

    tok_budget, done, ooms = 4096, 0, 0
    with torch.inference_mode():
        while done < len(order):
            end = done + 1
            while end < len(order) and (end - done + 1) * len(ids[order[end]]) <= tok_budget and (end - done) < 16:
                end += 1
            idx = order[done:end]
            try:
                width = max(len(ids[i]) for i in idx)
                inp = torch.full((len(idx), width), pad, dtype=torch.long)
                att = torch.zeros((len(idx), width), dtype=torch.long)
                for r, i in enumerate(idx):
                    inp[r, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
                    att[r, :len(ids[i])] = 1
                hidden = model(input_ids=inp.to(embed_dev), attention_mask=att.to(embed_dev), use_cache=False).last_hidden_state
                vec = hidden[torch.arange(len(idx), device=hidden.device), (att.sum(1) - 1).to(hidden.device)].float()
                emb[idx] = F.normalize(vec, dim=-1).cpu().numpy().astype(np.float16)
                done = end
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                ooms += 1
                tok_budget = max(len(ids[order[done]]), tok_budget // 2)
    assert np.isfinite(emb).all()
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Qwen3-Embedding-8B] shape={emb.shape}, ooms={ooms}, {time.time() - t0:.1f}s", flush=True)
    return emb


_GAUSS_MASK_CACHE = {}


def _gauss_lse(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    if keepdims:
        return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())


def _gauss_lw(x):
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(x, assume_centered=True))


def gauss_fit(x, y, prompts, d=256, temp=3.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, d_raw = x.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True)
    g = len(pu)
    mx = x.mean(0)
    sx = x.std(0)
    sx = np.where(sx > 1e-8, sx, 1.0)
    xs = (x - mx) / sx

    pm = np.zeros((g, d_raw), dtype=np.float64)
    np.add.at(pm, pinv, xs)
    pm /= np.bincount(pinv)[:, None]
    r_mat = xs - pm[pinv]
    d_proj = int(min(d, n - g - 1, d_raw))
    w_eig, u_eig = np.linalg.eigh(r_mat @ r_mat.T)
    order = np.argsort(w_eig)[::-1][:d_proj]
    w_eig, u_eig = np.clip(w_eig[order], 1e-12, None), u_eig[:, order]
    v_proj = (r_mat.T @ u_eig) / np.sqrt(w_eig)

    z = xs @ v_proj
    p_mat = np.zeros((g, d_proj), dtype=np.float64)
    cnt_p = np.bincount(pinv, minlength=g).astype(np.float64)
    mu = np.zeros((18, d_proj), dtype=np.float64)
    for _ in range(4):
        res = z - p_mat[pinv]
        for c_i in range(18):
            mu[c_i] = res[y == c_i].mean(0)
        res = z - mu[y]
        p_mat.fill(0.0)
        np.add.at(p_mat, pinv, res)
        p_mat /= cnt_p[:, None]

    e_res = z - mu[y] - p_mat[pinv]
    dof = max(n - g - 17, 1)
    se_raw = e_res.T @ e_res / dof
    ae = _gauss_lw(e_res)
    se = (1.0 - ae) * se_raw + ae * (np.trace(se_raw) / d_proj) * np.eye(d_proj)

    pc = p_mat - p_mat.mean(0)
    sp_raw = pc.T @ pc / max(g - 1, 1) - np.mean(1.0 / cnt_p) * se_raw
    ap = _gauss_lw(pc)
    tr = max(np.trace(sp_raw), 1e-8 * np.trace(se))
    sp = (1.0 - ap) * sp_raw + ap * (tr / d_proj) * np.eye(d_proj)

    chol = np.linalg.cholesky(se)
    chol_inv = np.linalg.inv(chol)
    m_mat = 0.5 * (chol_inv @ sp @ chol_inv.T + (chol_inv @ sp @ chol_inv.T).T)
    lam, q_mat = np.linalg.eigh(m_mat)
    lam = np.clip(lam, 1e-4, None)
    a_mat = q_mat.T @ chol_inv
    return {"mx": mx, "sx": sx, "W": v_proj @ a_mat.T, "m": mu @ a_mat.T, "lam": lam, "T": float(temp)}


def gauss_scores(x_pool, lab, prm):
    u = ((np.asarray(x_pool, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]
    lab = np.asarray(lab)
    m, lam, temp = prm["m"], prm["lam"], prm["T"]
    out = np.empty((len(u), 18), dtype=np.float64)
    sizes = np.bincount(lab)
    s1 = sizes[lab] == 1
    if s1.any():
        r1 = u[s1][:, None, :] - m[None]
        out[s1] = log_softmax(-0.5 * (r1 ** 2 / (1.0 + lam)).sum(-1) / temp, axis=1)
    for g in np.where(sizes >= 2)[0]:
        ii = np.where(lab == g)[0]
        ug = u[ii]
        ng = len(ug)
        if ng <= 3:
            gam = lam / (1.0 + ng * lam)
            rg = ug[:, None, :] - m[None]
            a_t = -0.5 * (rg ** 2 * (1.0 - gam)).sum(-1)
            g_t = rg * np.sqrt(gam)
            b_t = np.einsum("icd,jed->ijce", g_t, g_t)
            q_tab = np.zeros([18] * ng, dtype=np.float64)
            for i in range(ng):
                sh = [1] * ng
                sh[i] = 18
                q_tab = q_tab + a_t[i].reshape(sh)
            for i, j in itertools.combinations(range(ng), 2):
                sh = [1] * ng
                sh[i] = 18
                sh[j] = 18
                q_tab = q_tab + b_t[i, j].reshape(sh)
            if (ng, 18) not in _GAUSS_MASK_CACHE:
                grids = np.meshgrid(*[np.arange(18)] * ng, indexing="ij")
                bad = np.zeros([18] * ng, dtype=bool)
                for i, j in itertools.combinations(range(ng), 2):
                    bad |= grids[i] == grids[j]
                _GAUSS_MASK_CACHE[(ng, 18)] = bad
            qt = np.where(_GAUSS_MASK_CACHE[(ng, 18)], -np.inf, q_tab / temp)
            lz = _gauss_lse(qt)
            for i in range(ng):
                ax = tuple(k for k in range(ng) if k != i)
                out[ii[i]] = (_gauss_lse(qt, axis=ax) if ax else qt) - lz
        else:
            v = lam / (1.0 + (ng - 1) * lam)
            lg = log_softmax((-0.5 * ((ug[:, None, :] - m[None]) ** 2 / (1.0 + lam)).sum(-1)) / temp, axis=1)
            q = np.exp(lg)
            for _ in range(5):
                s = ug - q @ m
                pm = (s.sum(0)[None] - s) * v
                lg = -0.5 * ((ug[:, None, :] - pm[:, None, :] - m[None]) ** 2 / (1.0 + v)).sum(-1) / temp
                lo = np.log(np.clip(1.0 - q, 1e-6, None))
                lg = lg + (lo.sum(0)[None] - lo)
                q = np.exp(log_softmax(lg, axis=1))
            out[ii] = log_softmax(lg, axis=1)
    return out


def main():
    t0 = time.time()
    deadline_mono = time.monotonic() + DEADLINE_H * 3600.0
    print("=== Exp #016: 7-Backbone + Gauss Head (Llama3-S1 + Qwen3-8B-Base) + Qwen3-Emb Clusters + Global Sinkhorn(0.30) + GLM ===", flush=True)

    train = pd.read_csv(find_file("train.csv", path_hint="competitions/name-that-llm"))
    test = pd.read_csv(find_file("test.csv", path_hint="competitions/name-that-llm"))
    test["text"] = test["text"].fillna("").astype(str)
    classes = sorted(train["model"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y_train = train["model"].map(c2i).values
    pids_train = train["prompt_id"].values

    qwen_emb_dir = find_qwen_emb_dir()
    gauss_assets_dir = str(Path(find_file("Z_llama_s1.npy", path_hint="exp016-gauss-assets")).parent)
    llama8b_lora = find_qwen_lora_dir(4096, "exp015")
    llama8b_base = find_qwen_base_dir(4096, "llama")
    qwen8b_lora = find_qwen_lora_dir(4096, "exp013", required=False)
    qwen8b_base = find_qwen_base_dir(4096, "qwen") if qwen8b_lora is not None else None
    qwen4b_base = find_qwen_base_dir(2560, "qwen")
    qwen4b_lora = find_qwen_lora_dir(2560, "exp007")
    qwen17b_base = find_qwen_base_dir(2048, "qwen")
    qwen17b_lora = find_qwen_lora_dir(2048, "exp006")
    mb_large = find_encoder_dir(1024, "modernbert", "exp008")
    deb_large = find_encoder_dir(1024, "deberta-v2", "exp012")
    mb_base = find_encoder_dir(768, "modernbert", "exp005")
    stylegram_v1_dir = str(Path(find_file("bundle_meta.json", path_hint="exp011-stylegram")).parent)
    stylegram_v2_dir = str(Path(find_file("bundle_meta.json", path_hint="exp013-stylegram-v2")).parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    cap_major = torch.cuda.get_device_capability(0)[0] if use_amp else 0
    amp_dtype = torch.bfloat16 if (use_amp and cap_major >= 8) else torch.float16
    print(f"[DEVICE] {device} (count={torch.cuda.device_count()}) | cap_major={cap_major} | AMP={amp_dtype}", flush=True)

    texts = test["text"].values.tolist()
    G_emb = None
    if qwen_emb_dir is not None:
        E_test = extract_qwen_emb8b(texts, qwen_emb_dir, amp_dtype, use_amp)
        if len(texts) >= 40:
            sim_q = E_test.astype(np.float32) @ E_test.astype(np.float32).T
            G_emb = np.unique(capped_agglo(sim_q, cap=18, thr=QWEN_EMB_THR), return_inverse=True)[1]
            _, cnt_q = np.unique(G_emb, return_counts=True)
            print(f"[QWEN-EMB CLUSTERS] n={len(texts)}, clusters={len(cnt_q)}, grp2_18={(cnt_q[G_emb] >= 2).mean():.3f}, max={cnt_q.max()}", flush=True)
            del sim_q
        else:
            G_emb = np.arange(len(texts))
        del E_test
        gc.collect()

    s1_lora_path = os.path.join(llama8b_lora, "stage1_best", "lora_s1.pt")
    p_ll, z_ll_s1 = predict_qwen3_multilayer_ensemble(
        texts, classes, llama8b_base, llama8b_lora, amp_dtype, use_amp, 16, "LLAMA3-8B", extract_stage1_lora=s1_lora_path
    )
    p8, z_q8_base = predict_qwen3_multilayer_ensemble(
        texts, classes, qwen8b_base, qwen8b_lora, amp_dtype, use_amp, 16, "QWEN3-8B", extract_base_feats=True
    )
    p4 = predict_qwen3_ensemble(texts, classes, qwen4b_base, qwen4b_lora, device, amp_dtype, use_amp, 24, "QWEN3-4B")
    p_mbl = predict_modernbert_ensemble(texts, classes, mb_large, device, amp_dtype, use_amp, 48, "ModernBERT-large")
    p_deb = predict_deberta_multilayer_ensemble(texts, classes, deb_large, device, 32, "DeBERTa-v3-large")
    p17 = predict_qwen3_ensemble(texts, classes, qwen17b_base, qwen17b_lora, device, amp_dtype, use_amp, 32, "QWEN3-1.7B")
    p_mbb = predict_modernbert_ensemble(texts, classes, mb_base, device, amp_dtype, use_amp, 64, "ModernBERT-base")

    sg2_lp, gates_lp, families, G_tfidf = predict_stylegram_v2_and_clusters(texts, classes, stylegram_v2_dir)
    G_clust = G_emb if G_emb is not None else np.unique(G_tfidf, return_inverse=True)[1]
    sg1_lp = predict_stylegram_v1_with_clusters(texts, classes, stylegram_v1_dir, G_clust)

    t_g = time.time()
    prm_llama = gauss_fit(np.load(os.path.join(gauss_assets_dir, "Z_llama_s1.npy")), y_train, pids_train, d=GAUSS_D, temp=GAUSS_TEMP)
    lp_g_ll = gauss_scores(z_ll_s1, G_clust, prm_llama)
    del prm_llama, z_ll_s1
    gc.collect()
    prm_q8 = gauss_fit(np.load(os.path.join(gauss_assets_dir, "Z_qwen8b_base.npy")), y_train, pids_train, d=GAUSS_D, temp=GAUSS_TEMP)
    lp_g_q8 = gauss_scores(z_q8_base, G_clust, prm_q8)
    del prm_q8, z_q8_base
    gc.collect()
    gauss_lp = 0.5 * lp_g_ll + 0.5 * lp_g_q8
    assert np.isfinite(gauss_lp).all(), "Non-finite Gauss scores!"
    print(f"[GAUSS HEAD] Two-Way covariance + joint inference complete in {time.time() - t_g:.1f}s", flush=True)

    neu_lp = (
        W_QWEN3_8B * np.log(p8 + EPS)
        + W_QWEN3_4B * np.log(p4 + EPS)
        + W_MB_LARGE * np.log(p_mbl + EPS)
        + W_DEB_LARGE * np.log(p_deb + EPS)
        + W_QWEN3_17B * np.log(p17 + EPS)
        + W_MB_BASE * np.log(p_mbb + EPS)
        + W_LLAMA3_8B * np.log(p_ll + EPS)
    )

    bl = log_softmax(neu_lp + W_STYLEGRAM_V1 * sg1_lp + W_STYLEGRAM_V2 * sg2_lp + W_GAUSS * gauss_lp, axis=1)
    bl = apply_conditional_family_gates(bl, gates_lp, families, lam=FAMILY_GATE_LAM)
    if len(texts) >= 40:
        if CLUSTER_SK_ALPHA > 0:
            bl = cluster_sinkhorn(bl, G_clust, n_iters=3, alpha=CLUSTER_SK_ALPHA, lo=6, hi=18)
        bl = global_sinkhorn(bl, mix=GLOBAL_SK_MIX)

    pred_idx = bl.argmax(1)
    sub = pd.DataFrame({"id": test["id"], "model": [classes[i] for i in pred_idx]})
    sub.to_parquet("submission.parquet", index=False)
    print(f"[PART 1+2 DONE] Saved fallback submission.parquet in {time.time() - t0:.1f}s", flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    after_idx = run_glm_stage(test, bl, classes, deadline_mono)
    if after_idx is not None:
        sub = pd.DataFrame({"id": test["id"], "model": [classes[i] for i in after_idx]})
        sub.to_parquet("submission.parquet", index=False)

    sub_check = pd.read_parquet("submission.parquet")
    assert sub_check.columns.tolist() == ["id", "model"]
    assert len(sub_check) == len(test)
    assert sub_check["id"].astype(str).tolist() == test["id"].astype(str).tolist()
    assert not sub_check.isna().any().any()
    assert sub_check["model"].isin(classes).all()
    print(f"[DONE] Final submission.parquet verified shape={sub_check.shape} in {(time.time() - t0)/60:.1f}m", flush=True)
    print(sub_check.head(), flush=True)


if __name__ == "__main__":
    main()
