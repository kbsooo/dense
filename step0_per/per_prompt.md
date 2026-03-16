# PER Measurement Prompt

아래 프롬프트를 LLM에 그대로 붙여넣어서 결과를 받아오면 됨.
결과는 `per_results/` 폴더에 JSON으로 저장 (파일명 = 사용한 모델명).

---

## 프롬프트 (한국어 문장용)

```
아래 문장들을 각각 "풀어쓰기" 해주세요.

규칙:
1. 문장에 담긴 모든 암묵적 의미, 전제, 함축을 명시적으로 드러낼 것
2. 중학생이 이해할 수 있을 만큼 쉽고 자세하게 풀어쓸 것
3. 원문의 뜻을 빠짐없이 전달할 것 (의미 누락 금지)
4. 불필요한 수사나 반복 없이, 순수하게 의미를 전개할 것
5. 반드시 아래 JSON 형식으로만 답할 것 (다른 설명 없이)

입력 문장:
[KO_H01] 상부에서 자신들의 권위로 균형 잡힌 정의를 조성해주지 않는데도 다른 사람들의 권리를 자발적으로 존중하는 것은 그리 쉬운 일이 아니다
[KO_H02] 혁명이 숙적을 주인으로 만들지 않는다면 그것만으로도 행운이다
[KO_H03] 자유란 책임을 의미한다 그것이 대부분의 사람들이 자유를 두려워하는 이유다
[KO_H04] 인간의 모든 불행은 단 하나의 사실에서 비롯된다 그것은 자기 방에 조용히 머물 줄 모른다는 것이다
[KO_H05] 가장 긴 여행은 내면으로 향하는 여행이며 자기 자신을 선택한 자만이 비로소 타인에게 다가갈 자격을 얻는다
[KO_L01] 오늘 아침에 일찍 일어나서 따뜻한 물로 세수를 하고 밥을 먹은 다음에 학교에 갔다가 저녁에 집으로 돌아왔다
[KO_L02] 그 사람은 어제 서울에서 부산까지 기차를 타고 갔다가 오늘 다시 돌아왔다
[KO_L03] 나는 어제 친구를 만나서 같이 카페에 가서 커피를 마시고 이야기를 나눴다
[KO_L04] 주말에 가족들과 함께 근처 공원에 가서 산책을 하고 벤치에 앉아서 도시락을 먹었다
[KO_L05] 아침에 지하철을 타고 회사에 갔다가 점심에 근처 식당에서 밥을 먹고 오후에 다시 일했다

출력 형식 (이 JSON만 출력, 다른 텍스트 없이):
{
  "KO_H01": "풀어쓴 텍스트",
  "KO_H02": "풀어쓴 텍스트",
  "KO_H03": "풀어쓴 텍스트",
  "KO_H04": "풀어쓴 텍스트",
  "KO_H05": "풀어쓴 텍스트",
  "KO_L01": "풀어쓴 텍스트",
  "KO_L02": "풀어쓴 텍스트",
  "KO_L03": "풀어쓴 텍스트",
  "KO_L04": "풀어쓴 텍스트",
  "KO_L05": "풀어쓴 텍스트"
}
```

---

## 프롬프트 (영어 문장용)

```
Rewrite each of the following sentences so that ALL implicit meanings, presuppositions,
and implications are made fully explicit.

Rules:
1. Unpack every piece of assumed knowledge and hidden context
2. Write so a 14-year-old with no specialized background can understand fully
3. Preserve the complete meaning — no omissions allowed
4. No unnecessary repetition or filler — just clear, explicit meaning
5. Output ONLY the JSON below, no other text

Input sentences:
[EN_H01] The only thing we have to fear is fear itself — nameless, unreasoning, unjustified terror which paralyzes needed efforts to convert retreat into advance.
[EN_H02] He who fights with monsters should look to it that he himself does not become a monster, for when you gaze long into an abyss, the abyss also gazes into you.
[EN_H03] Power tends to corrupt and absolute power corrupts absolutely; great men are almost always bad men, even when they exercise influence and not authority.
[EN_H04] The arc of the moral universe is long, but it bends toward justice — though only when enough hands reach up to pull it down.
[EN_H05] In the middle of difficulty lies opportunity, but recognizing it demands the courage to abandon the certainty of what has already failed.
[EN_L01] I woke up early this morning, brushed my teeth, got dressed, ate breakfast at the kitchen table, and then drove to work in my car.
[EN_L02] She walked to the store on the corner, bought a bottle of water and a sandwich, then sat on a bench in the park and ate her lunch.
[EN_L03] Yesterday afternoon my brother came over to my house, we watched a movie on the couch, ordered pizza for dinner, and he went home around nine.
[EN_L04] This morning I took the bus to the library, returned three books that were due, checked out two new ones, and came back home before lunch.
[EN_L05] After work I stopped at the grocery store, picked up some milk and eggs, drove home, cooked a simple dinner, and watched the news on television.

Output ONLY this JSON, nothing else:
{
  "EN_H01": "expanded text",
  "EN_H02": "expanded text",
  "EN_H03": "expanded text",
  "EN_H04": "expanded text",
  "EN_H05": "expanded text",
  "EN_L01": "expanded text",
  "EN_L02": "expanded text",
  "EN_L03": "expanded text",
  "EN_L04": "expanded text",
  "EN_L05": "expanded text"
}
```

---

## 결과 저장 방법

받은 JSON을 `step0_per/per_results/` 폴더에 저장:

```
per_results/
├── claude-opus-4.json       # Claude Opus 4 결과
├── gpt-4o.json              # GPT-4o 결과
├── gemini-2.0-flash.json    # Gemini 결과
└── ...
```

파일명은 사용한 모델명으로 자유롭게. 여러 모델로 테스트할수록 PER의 일관성 검증이 강해짐.

---

## 주의사항

- 한국어 프롬프트와 영어 프롬프트를 **별도로** 전송하는 게 깔끔함
- 결과 JSON 두 개를 하나의 파일에 합쳐서 저장해도 됨:
  ```json
  {
    "KO_H01": "...",
    ...
    "EN_H01": "...",
    ...
  }
  ```
