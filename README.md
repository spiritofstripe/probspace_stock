# 1.概要
このページでは、ProbSpace主催のデータ分析コンペ「米国株式市場　将来株価予測」の、データと解法を掲載しています。
https://comp.probspace.com/competitions/us_stock_price

# 2.課題内容
2011/11/13～2019/11/17週の計419週間にわたる米国株データをもとに、2019/11/24週の終値を予測する。
外部データの使用は禁止。

# 3.当ページの構成
◆コンペ提供データ(company_list.csv , submission_template.csv , train_data.csv)

◆スクリプト(株価_前処理.R , 本体.R)

自分のローカル環境が貧弱なため、株価_前処理.Rで株価部分だけ先に前処理を実施し、一時保存フォルダに格納、
その後、本体.Rで他の前処理をして、株価_前処理.Rの内容を合体させています。

◆その他(LICENSE(MITライセンスの記載) , READNE.md)

# 4.分析に用いるフォルダ構成
(以下3つのフォルダを事前に作成)

任意のフォルダ　--　原本

　　　　　　　　|
        
　　　　　　　　-- 一時保存
        
　　　　　　　　|
        
　　　　　　　　-- 出力
# 5.使用モデル
LighrGBM 単一モデル

# 6.目的変数の変更
課題内容は株価予測ですが、そのままでは予測が難しく、また2019/11/24週のみの予測なので、
株価を対数変換し、1期前からの差分を目的変数としました。

# 7.主な特徴量
・年・月・年の始まりからの週数・四半期

・4・13・26・52週間隔での移動平均・移動標準偏差

・一定値以上(or以下)の株価へのフラグ付与と、その累積値

・一定期間における目的変数がプラス(orマイナス)である割合、増減回数

# 8.最後に言い訳
本職は非エンジニアでして、今回初めてgithubアカウントを作りました。
こういうオープンソース化も、初体験です。
いろいろ見苦しい点があるかと思いますが、何卒ご了承いただければ幸いでございます。
