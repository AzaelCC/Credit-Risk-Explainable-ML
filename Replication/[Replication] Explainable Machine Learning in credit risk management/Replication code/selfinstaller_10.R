rm(list=ls())

print("start selfinstaller")

options(rgl.useNULL = TRUE)
options(repos = c(CRAN = "http://mran.revolutionanalytics.com/snapshot/2020-02-01"))

used_packages <- c("ggpubr", "caTools", "SHAPforxgboost", "grDevices", "e1071", "tidyverse", "readxl", "caret", "xgboost", "ROCR", "mlr", "mlrMBO", "devtools","rgenoud", 
                   "emstreeR", "igraph", "graphlayouts", "grid")

for(package in used_packages) {
  if(!package %in% installed.packages()){
    install.packages(package)
  } 
}

devtools::install_github("ja-thomas/autoxgboost")
library("devtools")
library("autoxgboost") #devtools::install_github("ja-thomas/autoxgboost")


library("ggpubr")
library("caTools")
library("grDevices")
library("e1071")
library("tidyverse")
library("readxl")
library("caret")
library("xgboost")
library("ROCR")
library("mlr")
library("mlrMBO")
#library("devtools")
library("rgenoud")
library("emstreeR")
library("igraph")
library("graphlayouts")
library("grid")
library("autoxgboost") #devtools::install_github("ja-thomas/autoxgboost")
library("SHAPforxgboost")




print(sessionInfo())

print("selfinstaller finished!")



# Use Case 3:
#install.packages("microbenchmark")
#install.packages("optimx")
#install.packages("bigmemory")
# install.packages("XML");library(XML)# 
# install.packages("rgl")# 
# install.packages("BiocManager")
# BiocManager::install("genefilter", version = "3.8");library(genefilter)
# BiocManager::install("annotate", version = "3.8")# 
# install.packages("modeest");library(modeest)# 
# install.packages("shiny");library(shiny)
# install.packages("R.utils");library(R.utils)
# install.packages("shinyAce");library(shinyAce)
# install.packages("shinyjs");library(shinyjs)
# install.packages("rmarkdown");library(rmarkdown)
# install.packages("knitr");library(knitr)
# install.packages("shinyjqui");library(shinyjqui)# 
# install.packages("DescTools");library(DescTools)
# install.packages("ggplot2");library(ggplot2)
# install.packages("dplyr");library(dplyr)
# install.packages("readxl");library(readxl)
# install.packages("purrr");library(purrr)
# install.packages("Hmisc");library(Hmisc)
# install.packages("pastecs");library(pastecs)
# install.packages("ggpubr");library(ggpubr)
# install.packages("nortest");library(nortest)

# install.packages("clusterSim");library(clusterSim)
# install.packages("tidyr");library(tidyr)
# install.packages("reshape");library(reshape)
# install.packages("reporttools");library(reporttools)
# install.packages("naniar");library(naniar)
# install.packages("corrplot");library(corrplot)
# install.packages("corrgram");library(corrgram)
# install.packages("qgraph");library(qgraph)
# install.packages("networkD3");library(networkD3)
# 
# install.packages("ggplot2");library(ggplot2)
# install.packages("plotris");library(plotrix)
# install.packages("caret");library(caret)
# install.packages("ROCR");library(ROCR)
# install.packages("pROC");library(pROC)
# install.packages("rpart");library(rpart)
# install.packages("DescTools");library(DescTools)

# install.packages("randomForest");library(randomForest)
# install.packages("ggpubr");library(ggpubr)
# install.packages("MASS");library(MASS)







# # fuer Brankas Beispiel

# install.packages("ggrepel")
# install.packages("e1071")
# install.packages("ppcor")
# install.packages("cluster")
# install.packages("ape")
# install.packages("caret")
# install.packages("ClustOfVar")
# install.packages("RColorBrewer")
# install.packages("scales")
# install.packages("aod")
# 
#










#install.packages("viridis")








# library(shiny)
# library(R.utils)
# library(viridis)
# library(knitr)




# 
# library(DescTools)
# library(ggplot2)
# library(dplyr)
# library(readxl)
# library(Hmisc)
# library(pastecs)
# library(corrplot)
# library(ggpubr)
# library(ggrepel)
# #library(knitr)
# #library(readxl)
# library(e1071)
# library(ppcor)
# library(readr)
# #library(Hmisc)
# #library(pastecs)
# library(cluster)
# library(ape)
# library(caret)
# library(ClustOfVar)
# library(RColorBrewer)
# library(scales)
# library(aod)
# library(ROCR)
# library(pROC)
# library(rpart)
# library(rpart.plot)
# #library(corrplot)
# library(corrgram)
# library(randomForest)
# 
# library(shinyjs)
# library(shinyAce)







# install.packages("glue")  #wird irgendwie später für plotly usw. genutzt
# install.packages("circlize")  # wird igendwie für ComplexHeatmap benötigt
# install.packages("shiny")
# install.packages("igraph") # RUN apt-get install -y libcurl4-openssl-dev libxml2-dev libssl-dev
# install.packages("devtools"); library(devtools)
# install.packages("PortfolioAnalytics") #(R (??? 2.14.0), zoo, xts (??? 0.8), foreach, PerformanceAnalytics (??? 1.1.0))
# install.packages("/tmp/local_packages/Rdonlp2_3042.11.tar.gz", repos = NULL) #, type = "win.binary") #oder: nloptr: R interface to NLopt
# #install.packages("Rdonlp2", repos="http://R-Forge.R-project.org")
# #install.packages("C:/Users/JP/Documents/Webserver_Firamis/XAIVEST/dockertemplate/local_packages/Rdonlp2_3042.11.zip", repos = NULL, type = "win.binary")
# install.packages("ROI.plugin.quadprog")
# install.packages("plotly")
# install.packages("DT")
# install.packages("dygraphs")
# install_local("/tmp/local_packages/streamgraph-master") #devtools::install_github("hrbrmstr/streamgraph")
# #install.packages("C:/Users/JP/Documents/Webserver_Firamis/XAIVEST/dockertemplate/local_packages/streamgraph-master.zip", repos = NULL, type = "win.binary")
# install.packages("shinydashboard")
# install.packages("rintrojs")
# install.packages("shinyBS")
# install_local("/tmp/local_packages/katexR-master") #devtools::install_github("timelyportfolio/katexR")
# #install.packages("C:/Users/JP/Documents/Webserver_Firamis/XAIVEST/dockertemplate/local_packages/katexR-master.zip", repos = NULL, type = "win.binary")
# install.packages("shinyjs")
# install.packages("rhandsontable")
# install.packages("ineq")
# install.packages("d3heatmap")
# install.packages("phylocanvas")
# install.packages("ndtv")
# install.packages("visNetwork")
# install.packages("pryr")
# install.packages("rmarkdown")
# ##install.packages("doParallel")
# 
# install.packages("matrixcalc")
# install.packages("ade4")
# install.packages("shinythemes")
# ##install.packages("Rglpk")
# install.packages("FRAPO")
# install.packages("RiskPortfolios")
# install_local("/tmp/local_packages/mcrp-master")
# #install.packages("C:/Users/JP/Documents/Webserver_Firamis/XAIVEST/dockertemplate/local_packages/mcrp-master.zip", repos = NULL, type = "win.binary", ref = "master")
# 
# 
# install.packages("tawny")
# install_local("/tmp/local_packages/dashboardthemes-master")
# #install.packages("C:/Users/JP/Documents/Webserver_Firamis/XAIVEST/dockertemplate/local_packages/dashboardthemes-master.zip", repos = NULL, type = "win.binary")
# install.packages("data.table")
# install.packages("fastcluster");library("fastcluster")
# install.packages("dendextend");library("dendextend")
# install.packages("ape");library("ape")
# install.packages("Rtsne");library("Rtsne")
# install.packages("dbscan");library("dbscan")
# 
# install.packages("shinyaframe")
# #install.packages("rgl")
# #install.packages("rglwidget")
# install.packages("htmltools")
# 
# install.packages("shinyTree")
# install.packages("rportfolios")
# 
# install.packages("xlsx")
# install.packages("imputeTS")
# install.packages("stringr")
# install.packages("DEoptim")
# install.packages("timeSeries")
# 
# install.packages("doSNOW")
# install.packages("pvclust")
# 
# #install.packages("BiocManager")
# #install.packages("ComplexHeatmap")
# #install_github("jokergoo/ComplexHeatmap")
# 
# source("http://bioconductor.org/biocLite.R")
# biocLite("ComplexHeatmap")
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# library("shiny")
# library("igraph") # RUN apt-get install -y libcurl4-openssl-dev libxml2-dev libssl-dev
# library("devtools")
# library("PortfolioAnalytics") #(R (??? 2.14.0), zoo, xts (??? 0.8), foreach, PerformanceAnalytics (??? 1.1.0))
# library("Rdonlp2")
# library("ROI.plugin.quadprog")
# library("plotly")
# library("DT")
# library("dygraphs")
# library("streamgraph")
# library("shinydashboard")
# library("rintrojs")
# library("shinyBS")
# library("katexR")
# library("shinyjs")
# library("rhandsontable")
# library("ineq")
# library("d3heatmap")
# library("phylocanvas")
# library("ndtv")
# library("visNetwork")
# library("pryr")
# library("rmarkdown")
# 
# library("FRAPO")
# library("ade4")
# library("matrixcalc")
# library("shinythemes")
# library("RiskPortfolios")
# library("mcrp")
# library("dashboardthemes")
# library("tawny")
# library("data.table")
# 
# library("shinyaframe")
# #library(rgl)
# #library(rglwidget)
# library("htmltools")
# library("xts")
# 
# library("shinyTree")
# library("rportfolios")
# 
# 
# 
# library(xlsx)
# library(imputeTS)
# library(stringr)
# library(DEoptim)
# library(timeSeries)
# library(doSNOW)
# library(pvclust)
# library(ComplexHeatmap)




# install.packages("BiocManager")
# install.packages("ComplexHeatmap")
# 
# library(ComplexHeatmap)
# # if (!requireNamespace("BiocManager", quietly = TRUE))
# #   install.packages("BiocManager")
# # BiocManager::install("ComplexHeatmap", version = "3.8")
# print("finish selfinstaller1")
# 
# 
# install.packages("xlsx")
# install.packages("imputeTS")
# install.packages("stringr")
# install.packages("DEoptim")
# install.packages("timeSeries")
# 
# install.packages("doSNOW")
# install.packages("pvclust")
# 
# 
# 
# library(xlsx)
# library(imputeTS)
# library(stringr)
# library(DEoptim)
# library(timeSeries)
# library(doSNOW)
# library(pvclust)
# 
# 
# print("finish selfinstaller2")



