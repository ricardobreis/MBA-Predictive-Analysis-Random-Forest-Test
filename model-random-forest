################################################################################################
#
# MODELAGEM PREDITIVA - MBA Business Analytics e Big Data
# Por: RICARDO REIS
#
# PROVA - TELCO CHURN
#
################################################################################################

# LENDO OS DADOS
path <- "C:/Users/Ricardo/Documents/R-Projetos/TelcoChurn/"
base <- read.csv(paste(path,"dataset-telco-churn.csv",sep=""), 
                 sep=",",header = T,stringsAsFactors = T)[,-1] # deletando a primeira coluna
summary(base)
library("VIM")
matrixplot(base)
aggr(base)

# ANALISE BIVARIADA
# Variáveis quantitativas 
boxplot(base$Age ~ base$Churn)
boxplot(base$NumbLines   ~ base$Churn)
boxplot(base$CustTime   ~ base$Churn)
boxplot(base$MthIncome    ~ base$Churn)
boxplot(base$BillAmount ~ base$Churn)
boxplot(base$TimeCurAdrr ~ base$Churn)

#Variáveis qualitativas
prop.table(table(base$Churn))
prop.table(table(base$Region,   base$Churn),1)
prop.table(table(base$CableTV,       base$Churn),1)
prop.table(table(base$AutDebt, base$Churn),1)

################################################################################################
# AMOSTRAGEM DO DADOS
library(caret)
set.seed(12345)
index <- createDataPartition(base$Churn, p= 0.7,list = F)
data.train <- base[index, ] # base de desenvolvimento: 70%
data.test  <- base[-index,] # base de teste: 30%

# Checando se as proporções das amostras são próximas à base original
prop.table(table(base$Churn))
prop.table(table(data.train$Churn))
prop.table(table(data.test$Churn))

# Algoritmos de árvore necessitam que a variável resposta num problema de classificação seja 
# um factor; convertendo aqui nas amostras de desenvolvimento e teste
data.train$Churn <- as.factor(data.train$Churn)
data.test$Churn  <- as.factor(data.test$Churn)

################################################################################################
# MODELAGEM DOS DADOS - RANDOM FOREST

names  <- names(data.train) # salva o nome de todas as variáveis e escreve a fórmula
f_full <- as.formula(paste("Churn ~",
                           paste(names[!names %in% "Churn"], collapse = " + ")))
library(randomForest)
rndfor <- randomForest(f_full,data= data.train,importance = T, nodesize =200, ntree = 500)
rndfor

# Avaliando a evolução do erro com o aumento do número de árvores no ensemble
plot(rndfor, main= "Mensuração do erro")
legend("topright", c('Out-of-bag',"1","0"), lty=1, col=c("black","green","red"))

# Uma avaliação objetiva indica que a partir de ~50 árvores não há mais ganhos expressivos
rndfor2 <- randomForest(f_full,data= data.train,importance = T, nodesize =10, ntree = 50)
rndfor2

plot(rndfor2, main= "Mensuração do erro")
legend("topright", c('Out-of-bag',"1","0"), lty=1, col=c("black","green","red"))

# Importância das variáveis
varImpPlot(rndfor2, sort= T, main = "Importância das Variáveis")

# Aplicando o modelo nas amostras  e determinando as probabilidades
rndfor2.prob.train <- predict(rndfor2, type = "prob")[,2]
rndfor2.prob.test  <- predict(rndfor2,newdata = data.test, type = "prob")[,2]

# Comportamento da saida do modelo
hist(rndfor2.prob.test, breaks = 25, col = "lightblue",xlab= "Probabilidades",
     ylab= "Frequência",main= "Random Forest")
boxplot(rndfor2.prob.test ~ data.test$Churn,col= c("green", "red"), horizontal= T)

################################################################################################
# AVALIANDO A PERFORMANCE

# Métricas de discriminação para ambos modelos
library(hmeasure) 

rndfor.train  <- HMeasure(data.train$Churn,rndfor2.prob.train)
rndfor.test  <- HMeasure(data.test$Churn,rndfor2.prob.test)
rndfor.train$metrics
rndfor.test$metrics

library(pROC)

roc1 <- roc(data.test$Churn,rndfor2.prob.test)
y1 <- roc1$sensitivities
x1 <- 1-roc1$specificities

plot(x1,y1, type="n",
     xlab = "1 - Especificidade", 
     ylab= "Sensitividade")
lines(x1, y1,lwd=3,lty=1, col="purple") 
legend("topright", c('Random Forest'), lty=1, col=c("purple"))

################################################################################################
################################################################################################
