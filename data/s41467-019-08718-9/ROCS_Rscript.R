#Optimise unsupervised hierarchical clustering (Fig. 1d)
cor_cluster_all_new$cor_cluster_all=cutree(hclust(as.dist (1 - cor (t(scale(cor_cluster_all_new[,3:659])) / 2)),method="complete"),k=5)

#Filter cases with outlier thickness
cor_cluster_all_new_nocluster2=subset(cor_cluster_all_new,cor_cluster_all!=2)
cor_cluster_all_new_nocluster2=subset(cor_cluster_all_new_nocluster2,Thickness !=1.25 & Thickness != 3.2)
cor_cluster_all_new_nocluster2_HGS=subset(cor_cluster_all_new_nocluster2,Grade_codeBi !=0)

#Perform hierarchical clustering again using cleaned cases
cor_cluster_all_new_nocluster2$clusternew2=cutree(hclust(as.dist (1 - cor (t(scale(cor_cluster_all_new_nocluster2[,3:659])) / 2)),method="complete"),k=2)
plot(hclust(as.dist (1 - cor (t(scale(cor_cluster_all_new_nocluster2[,3:659])) / 2)),method="complete"),label=cor_cluster_all_new_nocluster2$clusternew2,main="hcluster by Pearson correlation")
plot(hclust(dist (((scale(cor_cluster_all_new_nocluster2[,3:659])))),method="complete"),label=cor_cluster_all_new_nocluster2$clusternew2,main="hcluster by Euclidean distance")

#Generate heatmap
library(heatmap.plus)
library(devtools)
install_github("fifer")
library(fifer)
#bilateralnew column indicates whether the two primaries are in the same cluster
heatmap.plus(t(scale(cor_cluster_all_new_nocluster2_HGS[,3:659],scale=T)),trace="none",col=my_palette, cexCol=0.1,cexRow=0.5,distfun=function(c) as.dist (1 - cor (t(c) / 2)),breaks=c(seq(-15,-3,length=101),seq(-3,-0.5,length=101)[c(-1)],seq(-0.5,0.5,length=101)[c(-1,-101)],seq(0.5,3,length=101)[c(-101)],seq(3,15,length=101)),labRow=FALSE,labCol=F,ColSideColors=cbind(string.to.colors(cor_cluster_all_new_nocluster2_HGS[,"bilateralnew"],colors=c("gold","green4","purple")),string.to.colors(cor_cluster_all_new_nocluster2_HGS[,"Ascites"],colors=c("grey39","lightcyan","chocolate")),string.to.colors(cor_cluster_all_new_nocluster2_HGS[,"StagecodeBi"],colors=c("grey","pink","grey39")),string.to.colors(cor_cluster_all_new_nocluster2_HGS[,"clusternew2"],c("blue","red"))),RowSideColors=cbind(Radiomic_featureclass[,"Classcolor"],Radiomic_featureclass[,"Classcolor"]),margins=c(10,14))
legend("topright",legend=c("Cluster 1","Cluster 2","","Stage I-II","Stage III-IV","NA","p= 0.0686","","Ascites","No ascites","p= 0.00729","","Close bilateral","Separate bilateral","Unilateral","96% bilateral tumours","clustered together"),
fill=c("blue","red","white","pink","grey","grey39","white","white","chocolate","lightcyan","white","white","gold","purple","green4","white","white"), border=F, bty="n", y.intersp = 0.8, cex=0.6)

#Generation of RPV using LASSO:
library(glmnet)
library(survival)
x=merge(cor_cluster_all_new_nocluster2, HH_ov_clinical,by.x="Patient_Idold",by.y="row.names")
x_complete=x[complete.cases(x[,c("Overall.survival..days.","OS.event")]),]
x_complete_scale=scale(x_complete[,3:659],scale=scalefactor$scale,center=scalefactor$center)

fit <- glmnet(x=as.matrix(x_complete_scale[,row.names(subset(coxosmulti_HH_adj,padj<0.05))]), y= Surv(x_complete $"Overall.survival..days.", x_complete $"OS.event"), family="cox",alpha=1)
plot(fit)
cvfit <- cv.glmnet(x=as.matrix(x_complete_scale[,row.names(subset(coxosmulti_HH_adj,padj<0.05))]), y= Surv(x_complete $"Overall.survival..days.", x_complete $"OS.event"), family="cox",alpha=1)
plot(cvfit)
fit <- glmnet(x=as.matrix(x_complete_scale[,row.names(subset(coxosmulti_HH_adj,padj<0.05))]), y= Surv(x_complete $"Overall.survival..days.", x_complete $"OS.event"), family="cox",alpha=1,lambda=cvfit$lambda.min)
fit$beta[,1]

#TCGA_ov_radiomics and HH_ov_radiomics are scaled and centered radiomic data, and include only primary tumours with higher RPV
#HH_ov_radiomics contains both the HH discovery and the HH validation datasets.
#Using the TCGA dataset as an example to calculate RPV based on weightings reported in the manuscript
RPV <- TCGA_ov_radiomics[,"FD_max_25HUgl"]*(-0.08764)+TCGA_ov_radiomics[,"GLRLM_SRLGLE_LLL_25HUgl"]*0.086901+TCGA_ov_radiomics[,"NGTDM_Contra_HLL_25HUgl"]*0.16509+TCGA_ov_radiomics[,"FOS_Imedian_LHH"]*0.250301

#Univariate and multivariable Cox regression analysis (Table 2):
library(survival)
summary(coxph(Surv(Overall.survival..days.,OS.event)~RPV,data= subset(TCGA_ov_clinical,Grade>1)))
summary(coxph(Surv(Overall.survival..days.,OS.event)~RPV+Stage+ age_at_initial_pathologic_diagnosis+ Surgery.outcome+Thickness,data= subset(TCGA_ov_clinical,Grade> 1)))

#Kaplan-Meier plot (Fig.2)
install.packages("devtools")
library(devtools)
install_github("michaelway/ggkm")
library(ggkm)
ggkm(survfit(Surv(Overall.survival..days.,OS.event)~cut(RPV,c(-10,0.095,0.658,10)),data= subset(TCGA_ov_clinical,Grade>1)),pval=T,table=T)

#To generate figures in Fig.3
#TCGA_clinical_molecular_combined is the dataframe containing RPV,eRPV, primary chemotherapy response molecular subtype, BRCA mutation, CCNE1 amplfication, stromal content, tumour content and RPPA data in the TCGA validation dataset
library(ggplot2)
library(reshape2)
RPV_patientfeatures_TCGA=TCGA_clinical_molecular_combined[,c("Row.names","RPV","ERPV","Response","SUBTYPE","BRCAdetail","CCNE1tri","stromalBi","tumorBi","Fibronectin.R.V","Rad51.R.V","FoxM1.R.V")]
x1<-melt(subset(RPV_patientfeatures_TCGA,RPV<10),id.var="Row.names",measure.var=c("RPV","ERPV"))
x1<-merge(x1,RPV_patientfeatures_TCGA[,c(1,2)],by.x=1,by.y=1)

#For barchart:
p1<- ggplot(x1,aes(reorder(Row.names,RPV),value,fill=variable))+geom_bar(stat= "identity")+theme(axis.text.x=element_blank(),axis.title.x=element_blank(),axis.ticks.x=element_blank(),legend.key.size = unit(0.3,"cm"),legend.key.width =unit(0.15,"cm"),panel.background =  element_rect(fill = NA))+
geom_bar(data=x1.1,stat= "identity")+
scale_fill_brewer(palette="Paired")+geom_line(group=1,colour="skyblue4")

#For categorical:
x2<-melt(subset(RPV_patientfeatures_TCGA,RPV<10),id.var="Row.names",measure.var=c("CCNE1cha","BRCAdetail","tumorBi","stromalBi","SUBTYPE","Response"))
x2<-merge(x2, RPV_patientfeatures_TCGA[,c(1,2)],by.x=1,by.y=1)
p2<- ggplot(x2, aes(x=reorder(Row.names,RPV), y=variable))+geom_tile(aes(fill = value),width=0.8, height= 0.8,colour="white") + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(),axis.text.x=element_blank(),axis.title.x=element_blank(),axis.ticks.x=element_blank(),legend.key.size = unit(0.3,"cm"),legend.key.width =unit(0.15,"cm")) +
  scale_fill_manual(values=c("#CCCCCC","#3399FF", "#CC99CC", "#CCCCCC", "#CCCCCC","#009999","#66CC66","#FF9933", "#FF663399","#FFCC66","#FFCC66","#CC0000","#CCCCCC","#003399","black","#CCCCCC"),na.value="grey95")

#For RPPA:
x3<-subset(RPV_patientfeatures_TCGA,RPV<10)[,c("Row.names","FoxM1.R.V","Rad51.R.V","Fibronectin.R.V")]
x3[,2:4]=scale(x3[,2:4])
x3<-melt(x3,id.var="Row.names",measure.var=c("FoxM1.R.V","Rad51.R.V","Fibronectin.R.V"))
x3<-merge(x3, RPV_patientfeatures_TCGA[,c(1,2)],by.x=1,by.y=1)
p3<- ggplot(x3, aes(x=reorder(Row.names,RPV), y=variable))+geom_tile(aes(fill = value),width=0.8, height= 0.8,colour="white") + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(),axis.text.x=element_blank(),axis.title.x=element_blank(),axis.ticks.x=element_blank(),legend.key.size = unit(0.3,"cm"),legend.key.width =unit(0.15,"cm")) +
  scale_fill_distiller(palette="RdYlBu" , na.value="grey95")

#multipanel
install.packages("ggpubr")
library(ggpubr)
ggarrange(p1 ,p3, p2 + font("x.text", size = 10),ncol = 1, nrow = 3,align = "v", heights=c(0.5,0.3,1))

