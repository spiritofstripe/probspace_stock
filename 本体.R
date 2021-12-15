
#目的変数を、株価そのものから、1期前からの対数差分に変更する。
#株価_前処理.Rで出力した結果を、途中で挿入する。

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
#事前準備(株価_前処理.Rと共通)
#############################################################################
#容量削減用
igc <- function() {
    invisible(gc()); invisible(gc())   
}


#入力
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


#############################################################################
#特徴量作成 - カレンダー関連とid付与
#############################################################################
year <- year(train$Date)
month <- month(train$Date)
week <- week(train$Date)

q <- rep(1,nrow(train))
q[month==4 | month==5 | month==6] <- 2
q[month==7 | month==8 | month==9] <- 3
q[month==10 | month==11 | month==12] <- 4


train <- cbind(1:nrow(train),train,year,month,week,q)
colnames(train)[1] <- "id"



#############################################################################
#特徴量作成 - company
#############################################################################
#NA処理
company$IPOyear[company$IPOyear=="n/a"] <- NA
company$IPOyear <- as.integer(company$IPOyear)

company$Sector[company$Sector=="n/a"] <- NA
company$Industry[company$Industry=="n/a"] <- NA

#NAであることのフラグを作成
IPOyear_na <- rep(0,nrow(company))
Sector_na <- rep(0,nrow(company))
Industry_na <- rep(0,nrow(company))

IPOyear_na[is.na(company$IPOyear)] <- 1
Sector_na[is.na(company$Sector)] <- 1
Industry_na[is.na(company$Industry)] <- 1

company <- cbind(company,IPOyear_na,Sector_na,Industry_na)


#Name,Symbolに重複がある
#重複の原因は、同一のName,SymbolにおいてListが複数存在する
#Listをone-hot encodingに変換する。

select <- select(company,Symbol,List)
select <- select %>% mutate(f=rep(1,nrow(select)))
select <- pivot_wider(select,names_from = "List", values_from = "f") %>% data.frame
select[is.na(select)] <- 0
select <- data.table(select)

company <- dplyr::left_join(company,select,"Symbol") %>% select(-List,-Name) %>% data.table %>% unique


#trainとcompanyの結合(trainはidと銘柄(x)のみ使用)
colnames(company)[which(colnames(company)=="Symbol")] <- "x"

train_company <- dplyr::left_join(select(train,id,x),company,"x") %>% data.table


#SectorとIndustryをcount encodingで変換する
count_names <- c("Sector","Industry")

select_count <- select(train_company,count_names) %>% data.frame

for(i in 1:length(count_names)){
   co_mat <- select_count[,i] %>% table %>% data.table
   colnames(co_mat) <- c(count_names[i],paste(count_names[i],"label",sep="_"))

   train_company <- dplyr::left_join(train_company,co_mat,count_names[i]) %>% data.table
}

train_company <- select(train_company,-count_names)

igc()


#############################################################################
#株価_前処理.Rの出力結果を挿入
#############################################################################
train_lagroll <- fread(file = "yourfolder/一時保存/lagroll.csv")　#lagroll

#lag_2,lag_3は不使用(2週・3週前の値を見て予測するケースは少ないと判断)
#specialは不使用(極端な値にフラグを立てているspecialは、当該週に極端な値であることを知っても遅く、test期間では必ずNAとなるので、不使用とする)

train_lagroll <- select(train_lagroll,-lag_2,-lag_3,-special)



#付与したid順に並び替え
train_lagroll <- train_lagroll[order(train_lagroll$id),]


#############################################################################
#全ての特徴量を結合
#############################################################################
all <- cbind(train,select(train_lagroll,-id),select(train_company,-id,-x))

invisible({rm(list=c("train"));gc();gc()}) #メモリ削除
invisible({rm(list=c("train_company"));gc();gc()}) #メモリ削除


test <- select(all[all$Date=="2019-11-24"],-y)
all <- all[all$Date<"2019-11-24"]



#############################################################################
#fold作成 
#############################################################################
#銘柄単位でgroupkfold

fold <- 5
random <- 999


unique_x <- unique(train$x) %>% sort


set.seed(random)
cv_folds <- groupKFold(unique_x, k = fold)


train_list <- list()
check_list <- list()
for(i in 1:fold){
    id_num <- as.numeric(unlist(cv_folds[i]))

    tid <- unique_x[id_num]
    cid <- setdiff(unique_x,tid)

    cy <- subset(all,all$x %in% cid) %>% select(-Date,-x)
    ty <- subset(all,all$x %in% tid) %>% select(-Date,-x)

    check_list <- c(check_list,list(cy))
    train_list <- c(train_list,list(ty))
}


test_list <- rep(list(select(test,-Date,-x)),fold)




############################################################################################
#lightgbmの準備
############################################################################################

dtrain_list <- lapply(train_list,function(x){
                      lgb.Dataset(as.matrix(select(x,-y,-id)),label = as.matrix(x$y)) })
dcheck_list <- lapply(check_list,function(x){
                      lgb.Dataset(as.matrix(select(x,-y,-id)),label = as.matrix(x$y)) })



watchlist <- list()
for(i in 1:length(train_list)){
    wl <- list(train = dtrain_list[[i]], eval = dcheck_list[[i]])
    watchlist <- c(watchlist,list(wl))
}

igc()


############################################################################################
#lightgbmの実行　パラメータチューニングはベイズ最適化
############################################################################################
init <- 10


bounds_lgb <- list(num_leaves = c(20L,100L),
                   max_depth = c(4L,12L),
                   bagging_fraction = c(0.001,1),
                   feature_fraction = c(0.001,1),
                   lambda_l1 = c(0,5),
                 　 lambda_l2 = c(0,5))



pb <- txtProgressBar(min = 1, max = length(train_list), style = 3)


lgb_param_list <- list()
imp_list <- list()
lgb_list <- list()
check_lgb_list <- list()
test_lgb_list <- list()
rmse_list <- c()
for(i in 1:length(train_list)){
   #model
   set.seed(random)
    lgb_bayes <- function(num_leaves,max_depth,bagging_fraction,feature_fraction,lambda_l1,lambda_l2) {
      param <- list(num_leaves = num_leaves,max_depth = max_depth,
                    bagging_fraction = bagging_fraction , feature_fraction = feature_fraction,
                    lambda_l1 = lambda_l1,lambda_l2 = lambda_l2,
                    boosting_type = "gbdt",objective = "regression", metric = "rmse")
        lgb <- lgb.train(param, dtrain_list[[i]], nrounds = 100,watchlist[[i]])
        pre_lgb <- predict(lgb,as.matrix(select(check_list[[i]],-id,-y)))
            
        rmse <- rmse(check_list[[i]]$y,pre_lgb)

        list(Score = -rmse,Pred = pre_lgb)
    }

   set.seed(random)
    lgb_OPT <- BayesianOptimization(lgb_bayes,bounds = bounds_lgb,init_points = init,
               n_iter = 1,acq = "ucb", kappa = 10, eps = 0.0,verbose = TRUE)

   set.seed(random)
    param_l <- list(learning_rate = 0.1, num_leaves = lgb_OPT$Best_Par[1], 
                    max_depth = lgb_OPT$Best_Par[2],
                    bagging_fraction = lgb_OPT$Best_Par[3], 
                    feature_fraction = lgb_OPT$Best_Par[4], 
                    lambda_l1 = lgb_OPT$Best_Par[5], 
                    lambda_l2 = lgb_OPT$Best_Par[6], 
                    boosting_type = "gbdt",objective = "regression", metric = "rmse")

   set.seed(random)
    lgb_model <- lgb.train(param_l, dtrain_list[[i]], nrounds = 5000,watchlist[[i]],verbose=1,
                           early_stopping_rounds=10,eval_freq=100)

   #predict
    check_lgb <- cbind(select(check_list[[i]],id,y),
                       predict(lgb_model,as.matrix(select(check_list[[i]],-id,-y))))
    colnames(check_lgb)[ncol(check_lgb)] <- "pre"   
    
    check_lgb <- check_lgb[order(check_lgb$id),]

    test_lgb <- predict(lgb_model,as.matrix(test_list[[i]][,-1]))
       

   #rmse
    rmse <- rmse(check_lgb$y,check_lgb$pre)
    

   #param
    lgb_param <- c(lgb_OPT$Best_Par,lgb_model$best_iter)
    names(lgb_param)[length(names(lgb_param))] <- "nrounds"

   #imp
   df_imp <- tbl_df(lgb.importance(lgb_model, percentage = TRUE))
   df_imp$Feature <- factor(df_imp$Feature, levels = rev(df_imp$Feature))


   setTxtProgressBar(pb, i) 

   #list化
   lgb_param_list <- c(lgb_param_list,list(lgb_param))
   imp_list <- c(imp_list,list(df_imp))
   lgb_list <- c(lgb_list,list(lgb_model))
   check_lgb_list <- c(check_lgb_list,list(check_lgb))
   test_lgb_list <- c(test_lgb_list,list(test_lgb))
   rmse_list <- c(rmse_list,rmse)
}


#パラメータ抽出
lgb_param_mat <- do.call("rbind",lgb_param_list) %>% data.table
param_score_mat <- cbind(lgb_param_mat,rmse_list)

#特徴量重要度抽出
imp_mat <- imp_list[[1]][,1:2]
for(i in 2:length(imp_list)){
    imp_mat <- dplyr::full_join(imp_mat,imp_list[[i]][,1:2],c("Feature"))
}


#validation用の整理
check_lgb_pre <- do.call("rbind",check_lgb_list) %>% data.table
check_lgb_pre <- check_lgb_pre[order(check_lgb_pre$id),]

#提出用の整理
test_lgb_pre <- do.call("cbind",test_lgb_list)
test_lgb_pre <- apply(test_lgb_pre,1,mean)




#対数差分から株価へ変換(validation)
check_0 <- dplyr::inner_join(select(train,id,Date,x),train_0,c("Date","x")) %>% 
           dplyr::inner_join(check_lgb_pre,c("id")) %>% data.table 
colnames(check_0) <- c("id","Date","x","y","y_lag","y_ratio_true","y_ratio_pre")


check_y_true <- check_0$y
check_y_pre <- check_0$y_lag + check_0$y_ratio_pre


check_y_true <- expm1(check_y_true)
check_y_pre <- expm1(check_y_pre)

#validationスコア
rmse(log1p(check_y_true),log1p(check_y_pre))


#対数差分から株価へ変換(提出用)
test_0 <- cbind(train_0[train_0$Date=="2019-11-17"],test_lgb_pre)

test_pre <- test_0$y + test_0$test_lgb_pre
test_pre <- expm1(test_pre)




####################################################################################################
#出力
####################################################################################################
submit$y <- test_pre

fwrite(submit, "yourfolder/出力/submit.csv")


