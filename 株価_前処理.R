

#目的変数を、株価そのものから、1期前からの対数差分に変更する。
#計算時のメモリ削減、時間短縮のため、このファイルでの前処理結果は、一時保存フォルダに保存する。

library(lightgbm)
library(Metrics)
library(data.table)
library(dplyr)
library(caret)
library(Matrix)
library(recipes)
library(stringr)
library(pbapply)
library(stringi)
library(tidyr)
library(MLmetrics)
library(rBayesianOptimization)
library(tcltk)
library(pbapply)
library(e1071)
library(lubridate)
library(zoo)
library(stats)


#############################################################################
#事前準備(本体.Rと共通)
#############################################################################
#容量削減用
igc <- function() {
    invisible(gc()); invisible(gc())   
}



#データ入力
setwd("yourfolder/原本")
train <- fread(file = "train_data.csv")
company <- fread(file = "company_list.csv")
submit <- fread(file = "submission_template.csv")

train$Date <- train$Date　%>% as.Date

#対数変換
train <- cbind(train[,1],log1p(train[,-1]))


#wide型からlong型へ型変更
#当該日の株価と1週前の株価を並列させる
train_0 <- train %>% pivot_longer(col = -Date, names_to = "x", values_to = "y") %>% data.table %>% na.omit

train_0_2 <- train[2:nrow(train),] %>% 
　　　　　　　　　　　　　pivot_longer(col = -Date, names_to = "x", values_to = "y") %>% data.table %>% na.omit　#1期前

train_0 <- cbind(train_0_2,select(train_0[train_0$Date!="2019-11-17"],y))
colnames(train_0) <- c("Date","x","y","y_lag")

invisible({rm(list=c("train_0_2"));gc();gc()})


#目的変数を対数差分に変換
#差分もwide型からlong型へ型変更
diff <- train[2:nrow(train),-1] - train[1:(nrow(train)-1),-1]

train <- cbind(train[-1,1],diff)

train <- train %>% pivot_longer(col = -Date, names_to = "x", values_to = "y") %>% data.table
train$Date[is.na(train$Date)] <- "2019-11-24"

train$Date <- train$Date　%>% as.Date


#id付与
train <- cbind(1:nrow(train),train)
colnames(train)[1] <- "id"


#############################################################################
#株価の特徴量作成
#A.lag / B.異常時 / C.平常時 / D.塊単位のlag / E.移動平均 の5種類

#A.lag --- 1-3期前の値を追加
#B.異常時 --- 極端に高い・低い株価に対してフラグを付与し、累積値や一定期間内の出現回数も追加
#C.平常時　---　一定期間内における、目的変数がプラスである・マイナスである回数の割合、目的変数の増加・減少の割合、「0」をまたいだ回数、一定期間内のNA個数を追加
#D.塊単位のlag --- 下記のroll_vecの期間単位に集約値を求めて、その集約値を4ブロック用意し、C.平常時と同じ特徴量を作成
#(D.例: 4＝4期の場合、1～4、5～8、9～12、13～16のlag値の平均をとることで、[1:1～4][2:5～8][3:9～12][4:13～16]の4ブロックを用意し、4ブロックを一定期間として、C.平常時の特徴量を作成する)
#E.移動平均 --- 下記のroll_vecの期間単位にmeanとsdを追加

#############################################################################

#銘柄別にリスト化して、銘柄単位の作業
train_list <- split(train,train$x)


#B～Eで使う、一定期間を設定(例.4は4週間)
roll_vec <- c(4,13,26,52)


lag_roll_func <- function(s){

   ########################
   #A.lag
   #lag作成関数
   lag_make_func <- function(vec,start,range){

      lag_short_mat <- shift(vec, n=start, fill=NA, type="lag")
      for(i in (start+1):range){
          lag <- shift(vec, n=i, fill=NA, type="lag")
          lag_short_mat <- cbind(lag_short_mat,lag)
      }

      lag_short_mat

   }


   #3期前までのlagを収集
   lag_short_mat <- lag_make_func(s$y,1,3)

   lag_short_mat <- data.table(lag_short_mat)
   colnames(lag_short_mat) <- paste("lag",1:3,sep="_")


   ########################
   #B.異常時
   #フラグの付け方
   #+0.3以上 == 1 , +0.5以上 == 2 , +0.7以上 == 3 , +1.0以上 == 4
   #-0.3以下 == -1 , -0.5以下 == -2 , -0.7以下 == -3 , -1.0以下 == -4

   sy <- s$y
   sy[is.na(sy)] <- 0

   #異常フラグ(後の本体.Rでこの特徴量は削除する/当該週に極端な値であることを知っても遅く、test期間では必ずNAとなるため)
   special <- rep(0,nrow(s))
   special[sy>=0.3] <- 1
   special[sy>=0.5] <- 2
   special[sy>=0.7] <- 3
   special[sy>=1.0] <- 4
   special[sy<= -0.3] <- -1
   special[sy<= -0.5] <- -2
   special[sy<= -0.7] <- -3
   special[sy<= -1.0] <- -4

   #異常累積値(異常の累積の大小を表現するために、異常フラグの1～4、-4～-1を累積和を使う)
　　　special_plus <- special 
   special_plus[special_plus<=0] <- 0
   special_plus <- cumsum(special_plus)

   special_minus <- special 
   special_minus[special_minus>=0] <- 0
   special_minus <- cumsum(special_minus)

   special[is.na(s$y)] <- NA

   
   #一定期間での異常値出現回数
   special_list <- list()
   for(i in 1:length(roll_vec)){
      af <- lag_make_func(special,1,roll_vec[i])

      af_minus <- apply(af,1,function(x){length(which(x<0))   })
      af_plus <- apply(af,1,function(x){length(which(x>0))   })

      af_mat <- cbind(af_minus,af_plus)
      colnames(af_mat) <- paste(colnames(af_mat),"roll",roll_vec[i],sep="_")

      special_list <- c(special_list,list(af_mat))

   }

   special_appear <- do.call("cbind",special_list) %>% data.table

   special_mat <- cbind(special,special_plus,special_minus,special_appear)


   ########################
   #C.平常時
　　　#roll_vecごとに、以下4つのlistを作成

   plus_minus_list <- list()　###　一定期間内のプラス・マイナスの出現割合
   updown_list <- list()　###　一定期間内の目的変数増加・減少の割合
   updown_0_list <- list()　###　一定期間内の「0」をまたいだ回数
   na_list <- list() ###　一定期間内のNA個数
   for(i in 1:length(roll_vec)){

      ssm <- lag_make_func(s$y,1,roll_vec[i])
      na_volume <- apply(ssm,1,function(x){length(which(is.na(x))) }) 

      na_list <- c(na_list,list(na_volume))

      #+-割合
      plus <- apply(ssm,1,function(x){length(which(x>0))  }) / roll_vec[i]
      minus <- apply(ssm,1,function(x){length(which(x<0))  }) / roll_vec[i]

      plus_minus <- data.table(plus,minus)
      colnames(plus_minus) <- paste(colnames(plus_minus),"ratio","roll",roll_vec[i],sep="_")

      plus_minus_list <- c(plus_minus_list,list(plus_minus))


      #updown
      ssm_diff <- ssm[,2:ncol(ssm)] - ssm[,1:(ncol(ssm)-1)]

      mo <- abs(apply(ssm_diff,1,function(x){length(which(is.na(x))) }) - (roll_vec[i] - 1))

      up_ratio <- apply(ssm_diff,1,function(x){length(which(x>0))   }) / mo
      down_ratio <- apply(ssm_diff,1,function(x){length(which(x<0))   }) / mo

      updown <- data.table(up_ratio,down_ratio)
      colnames(updown) <- paste(colnames(updown),"roll",roll_vec[i],sep="_")

      updown_list <- c(updown_list,list(updown))


      #0またぎ
      ssm_2 <- ssm
  
      ssm_2[which(ssm_2<0)] <- -1
      ssm_2[which(ssm_2>0)] <- 1

      ssm_diff_2 <- ssm_2[,2:ncol(ssm_2)] - ssm_2[,1:(ncol(ssm_2)-1)]

      up_0_ratio <- apply(ssm_diff_2,1,function(x){length(which(x>0))   }) / mo
      down_0_ratio <- apply(ssm_diff_2,1,function(x){length(which(x<0))   }) / mo

      updown_0 <- data.table(up_0_ratio,down_0_ratio)
      colnames(updown_0) <- paste(colnames(updown_0),"over","roll",roll_vec[i],sep="_")

      updown_0_list <- c(updown_0_list,list(updown_0))

   }

   plus_minus <- do.call("cbind",plus_minus_list) %>% data.table
   updown <- do.call("cbind",updown_list) %>% data.table
   updown_0 <- do.call("cbind",updown_0_list) %>% data.table
   na_volume <- do.call("cbind",na_list) %>% data.table

   colnames(na_volume) <- paste("na_volume","roll",roll_vec,sep="_")

   normal_phase <- cbind(plus_minus,updown,updown_0,na_volume)



   ########################
   #D.塊単位のlag
   
   lag_block_combi_list <- list()
   for(h in 1:length(roll_vec)){

      ########################
　　　　　　#塊単位の値を用意
      roll <- roll_vec[h]
      end <- seq(roll,roll*4,roll) ##　4つのブロックを作成
      start <- end - roll + 1 ##　4つのブロックを作成

　　　　　　#4つのブロックで平均値
      lag_block_list <- list()
      for(i in 1:length(end)){
         dm <- lag_make_func(s$y,start[i],end[i]) %>% apply(1,function(x){mean(x,na.rm=T)})
         lag_block_list <- c(lag_block_list,list(dm))
      }

      lag_block_mat <- do.call("cbind",lag_block_list)
      colnames(lag_block_mat) <- paste("lag",roll,"season",1:length(end),sep="_")


      ########################
　　　　　#以下、C.平常時と同じ特徴量の作成
      na_volume <- apply(lag_block_mat,1,function(x){length(which(is.na(x))) }) 

      #+-割合
      plus <- apply(lag_block_mat,1,function(x){length(which(x>0))  }) / roll
      minus <- apply(lag_block_mat,1,function(x){length(which(x<0))  }) / roll

      plus_minus <- data.table(plus,minus)
      colnames(plus_minus) <- paste(colnames(plus_minus),"ratio","lag_block",roll,sep="_")


      #updown
      lag_block_diff <- lag_block_mat[,2:ncol(lag_block_mat)] - lag_block_mat[,1:(ncol(lag_block_mat)-1)]

      mo <- abs(apply(lag_block_diff,1,function(x){length(which(is.na(x))) }) - (roll - 1))

      up_ratio <- apply(lag_block_diff,1,function(x){length(which(x>0))   }) / mo
      down_ratio <- apply(lag_block_diff,1,function(x){length(which(x<0))   }) / mo

      updown <- data.table(up_ratio,down_ratio)
      colnames(updown) <- paste(colnames(updown),"lag_block",roll,sep="_")


      #0またぎ
      lag_block_2 <- lag_block_mat
  
      lag_block_2[which(lag_block_2<0)] <- -1
      lag_block_2[which(lag_block_2>0)] <- 1

      lag_block_diff_2 <- lag_block_2[,2:ncol(lag_block_2)] - lag_block_2[,1:(ncol(lag_block_2)-1)]

      up_0_ratio <- apply(lag_block_diff_2,1,function(x){length(which(x>0))   }) / mo
      down_0_ratio <- apply(lag_block_diff_2,1,function(x){length(which(x<0))   }) / mo

      updown_0 <- data.table(up_0_ratio,down_0_ratio)
      colnames(updown_0) <- paste(colnames(updown_0),"over","lag_block",roll,sep="_")

      lag_block_all <- cbind(plus_minus,updown,updown_0,na_volume)
      colnames(lag_block_all)[ncol(lag_block_all)] <- paste("na_volume","lag_block",roll,sep="_")

      lag_block_combi_list <- c(lag_block_combi_list,list(lag_block_all))

   }

   lag_block_mat <- do.call("cbind",lag_block_combi_list) %>% data.table


   ########################
   #E.移動平均

   rollmean_list <- list()
   rollsd_list <- list()
   for(i in 1:length(roll_vec)){
       rollmean <- rollmean(s$y, roll_vec[i], fill=NA,align = c("right"))
       rollsd <- rollapplyr(s$y, roll_vec[i], sd,fill=NA,align = c("right"))

　　　　　　#目的変数を含んだ末尾のroll値を削除し、代わりに先頭の値をNAに補完する
       rollmean <- c(NA,rollmean[-length(rollmean)])
       rollsd <- c(NA,rollsd[-length(rollsd)])

       rollmean_list <- c(rollmean_list,list(rollmean))
       rollsd_list <- c(rollsd_list,list(rollsd))
   }

   rollmean_mat <- do.call("cbind",rollmean_list) %>% data.table
   rollsd_mat <- do.call("cbind",rollsd_list) %>% data.table

   colnames(rollmean_mat) <- paste("rollmean",roll_vec,sep="_")
   colnames(rollsd_mat) <- paste("rollsd",roll_vec,sep="_")


   lag_roll_mat <- cbind(lag_short_mat,special_mat,normal_phase,lag_block_mat,
                         rollmean_mat,rollsd_mat)


   s <- cbind(select(s,id),lag_roll_mat)

   s

}


train_list <- pblapply(train_list,lag_roll_func)

train_lagroll <- do.call("rbind",train_list) %>% data.table

invisible({rm(list=c("train_list"));gc();gc()}) #メモリ削除
igc()

####################################################################################################
#出力
fwrite(train_lagroll, "yourfolder/一時保存/lagroll.csv")


