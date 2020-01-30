# <br/> 웹툰 개인화 추천시스템(Naver Webtoon Recommendation System)
## Coala Univ 2기 해커톤 진행 (2020. 01. 11.)
### 중앙대 COCAUIN TEAM : "코알라유니브(COalauniv) 중앙대(CAU)인"

  
## 1. 코알라 유니브(Coala Univ)

[코알라 유니브(Coala Univ)](https://coalastudy.com/p/%EC%BD%94%EC%95%8C%EB%9D%BC%EC%9C%A0%EB%8B%88%EB%B8%8C)는 데이터 사이언스에 관심있는 대학생들을 모아 코딩 기초부터 데이터분석, 머신러닝 교육을 받을 수 있는 프로그램입니다.

코알라유니브 2기의 경우, 서울 14개 학교를 대상으로 진행되었고, 각 학교당 리더 2명과 학생 12명을 포함하여 총 14명을 선발하였습니다.

- [Coala Univ 홈페이지](https://coalastudy.com/)
- [Coala Univ 페이스북](https://www.facebook.com/coalastudy/)
- [Meeta 홈페이지](https://meeta.io/)

*관련기사*
- ['코딩좀알려주라', '코알라 유니브 2기' 모집](http://www.greenpostkorea.co.kr/news/articleView.html?idxno=108004?obj=Tzo4OiJzdGRDbGFzcyI6Mjp7czo3OiJyZWZlcmVyIjtOO3M6NzoiZm9yd2FyZCI7czoxMzoid2ViIHRvIG1vYmlsZSI7fQ%3D%3D)
- ['대학생을 데이터 사이언티스트로” 코알라 유니브 2기 모집'](https://search.naver.com/p/crd/rd?m=1&px=620&py=403&sx=620&sy=303&p=UBKD2wp0YidssidIXpNssssssrC-205593&q=%EC%BD%94%EC%95%8C%EB%9D%BC%EC%9C%A0%EB%8B%88%EB%B8%8C&ie=utf8&rev=1&ssc=tab.news.all&f=news&w=news&s=zZ5SZsAgLzO%2B0VjcWlLw9imd&time=1580410338699&bt=15&a=nws*f.tit&r=2&i=8817ca87_000000000000000000007543&g=5506.0000007543&u=https%3A%2F%2Fwww.venturesquare.net%2F788771)

## 2. 웹툰 개인화 추천시스템(Naver Webtoon Recommendation System)  
**설명 하단에 발표자료 포함**  
  
**[해당 해커톤 프로젝트 수행동기]**    
저희는 이번 기회로 추천시스템을 공부하고 구현해보는 것을 목적으로 진행하였습니다.  
  
**[데이터 수집]**  
먼저, [네이버 웹툰](https://comic.naver.com/webtoon/weekday.nhn) 중 **연재작 99, 완결작 50, 총 149개 작품**들을 대상으로, 크롤링한 정보(제목, 포스터, 작가)와 함께 설문조사폼을 만들습니다. 일주일 간 지인들에게 설문조사를 실시하여 약 270명의 웹툰 이용 기록 데이터를 수집할 수 있었습니다.  
  
**[사용 모델]**  
가능한 여러가지 모델을 이용하여 구현하고자 다음과 같은 모델들로 웹툰 개인화 기반 추천 시스템을 구현하였습니다.
    
- 잠재요소 기반 협업필터링(Latent factor based Collaborative Filtering)
- 아이템 기반 협업필터링(Item based Collaborative Filtering)
- 'Surprise' Module을 사용한 추천 시스템 모델('Surprise' based recommendation system)  
  
**[중요 파일]**  

- `00. 2기_중앙대COCAUIN(코카인)_01.웹툰추천시스템` : 추천시스템을 구현한 코드 파일입니다.  
- `01. 2기_중앙대COCAUIN(코카인)_02_Web Scraping for Survey_Form (NAVER Webtoon Service)` : 설문조사폼을 만들 때 활용한 크롤링 코드입니다. 웹툰 작품들의 제목, 포스터, 작가를 수집하였습니다.

## 3. Project Presentation

<img src = '/slides/slide1.PNG'>
<img src = '/slides/slide2.PNG'>
<img src = '/slides/slide3.PNG'>
<img src = '/slides/slide4.PNG'>
<img src = '/slides/slide5.PNG'>
<img src = '/slides/slide6.PNG'>
<img src = '/slides/slide7.PNG'>
<img src = '/slides/slide8.PNG'>
<img src = '/slides/slide9.PNG'>
<img src = '/slides/slide10.PNG'>
<img src = '/slides/slide11.PNG'>
<img src = '/slides/slide12.PNG'>
<img src = '/slides/slide13.PNG'>
<img src = '/slides/slide14.PNG'>
<img src = '/slides/slide15.PNG'>
