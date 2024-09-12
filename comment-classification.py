import openai
import pandas as pd
import time

try:
    Cfile = pd.read_csv("C:/Users/minth/인기급상승_20240324033044_commnet_total.csv", encoding='UTF8')
except:
    Cfile = pd.read_csv("C:/Users/minth/인기급상승_20240324033044_commnet_total.csv", encoding='cp949')

if len(Cfile.columns) <= 2:
    Cfile['Label'] = pd.Series([None] * len(Cfile))

openai.api_key = ""

idxText = ""
idxList = []
idx = 0

start = int(input("시작 인덱스를 적어주세요 : "))
end = int(input("마지막 인덱스를 적어주세요 : "))

if start > end:
    start, end = end, start

if end > len(Cfile):
    end = len(Cfile)
if start > len(Cfile):
    start = 0

for i in range(start, end):
    if Cfile.iloc[i, 2] == "" or Cfile.iloc[i, 2] == None or pd.isna(Cfile.iloc[i, 2]):
        idx += 1
        if idxText == "":
            idxText = "\""+str(Cfile.iloc[i, 1])+"\""
        else:
            try:
                idxText += "||" + "\""+str(Cfile.iloc[i, 1])+"\""
            except:
                idxText += "실패"
        idxList.append(i)

    if idx == 1:
        print("\r리퀘스트 보냄 : " + str(i) + "번째", end="")
        #print(Cfile.iloc[i, 1])
        chkSuc = True
        tryidx = 0 
        while chkSuc:
            if tryidx > 10:
                 for j in range(len(reqList)):
                        Cfile.iloc[idxList[j], 2] = -1
                 break
            try:
                tryidx += 1
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "당신은 댓글을 읽고 감정을 분석하는 AI입니다. 텍스트 입력이 들어오면 \"긍정\" 또는 \"부정\" 이라고 대답해주세요. 입력값은 || 으로 구분되고 "+str(len(idxList))+"개 입니다. 답변은 ||으로 띄어쓰기 없이 구분하여 답변해주세요."},
                        {"role": "user", "content": idxText}
                    ]
                )
                reqList = response.choices[0].message.content
                reqList = reqList.replace(" ","").split("||")
                if len(reqList) == len(idxList):
                    for j in range(len(reqList)):
                        if reqList[j] == "긍정":
                            Cfile.iloc[idxList[j], 2] = 1
                        elif reqList[j] == "부정":
                            Cfile.iloc[idxList[j], 2] = 0
                        else:
                            chkSuc = False
                            print(response.choices[0].message.content)
                            print(idxText)
                            print(" / 재시도", end = "")
                            time.sleep(10)
                    chkSuc = not chkSuc
                else:
                    print(" / 길이가 맞지 않음", end = "")

            except openai.OpenAIError as e:
                if isinstance(e, openai.BadRequestError):
                    for j in idxList:
                        Cfile.iloc[j, 2] = -1
                    chkSuc = not chkSuc
                else:
                    print(e, end = "")
                    print(idxText, end = "")
                    time.sleep(60)
        idxText = ""
        idxList = []
        idx = 0
        Cfile.to_csv("C:/Users/minth/인기급상승_20240324033044_commnet_total.csv", index = False, encoding='utf-8-sig')
Cfile.to_csv("C:/Users/minth/인기급상승_20240324033044_commnet_total.csv", index = False, encoding='utf-8-sig')
