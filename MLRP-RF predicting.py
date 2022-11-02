# -*- coding: utf-8 -*-
# @File   : MLRP-RF predicting
# @Time   : 2022/11/2 14:53 
# @Author : linjin😀

import numpy as np
import xlsxwriter
from sklearn.externals import joblib
from MLRP_RF_training import getControlID,getControlAntecedent1,getControlAntecedent2,getControlAntecedent3,getControlAntecedent4,getControlAntecedent5,Traningdata


'''predicting by forest models '''
def RFRpredicting(forest, test_data_only):
    """
    :param forest: The forest models achieveed from the MLRP_TRAINING
    :param test_data_only: The testing samples
    :return: Return the predictions 15 steps aheand by MLRP-RF of five controlled variables C1,C2,C3,C4,C5
    """
    ''' Predicting by tree model'''
    def treePredict(trees,onesample_only):
        """
        :param trees: The tree models in the form of dictionary
        :param onesample_only:Onesample without the tag data
        :return: Return the predictions by means of tree models
        """
        if(not isinstance(trees,dict)):#If not a dictionary then:
            return trees
        onesample_only_zip=np.zeros(np.shape(onesample_only)[1])#onesample_only_zip：1*10
        for j in range(np.shape(onesample_only)[1]):
            onesample_only_zip[j]=np.mean(onesample_only[:,j].astype('float'))
        if(onesample_only_zip[trees['best_dindex']]>trees['best_value']):#This sample is greater than the best value under the segmentation variable 'best_dindex'
            if(type(trees['left'])==list):
                return trees['left']
            else:
                return treePredict(trees=trees['left'],onesample_only=onesample_only)
        else:
            if(type(trees['right'])==list):
                return trees['right']
            else:
                return treePredict(trees=trees['right'],onesample_only=onesample_only)

    '''Conducting the predicting for the whole testing data'''
    def predictTestdtaByOneTree(trees,testdata_only):
        """
        :param trees: The tree models in the form of dictionary
        :param testdata_only: Testingsample without the tag data
        :return: Return the predictions of five conrolled variables of the whole testing dataset
        """
        C1list = []
        C2list = []
        C3list = []
        C4list = []
        C5list = []
        for i in range(len(testdata_only)):
            predictionsFiveVar=treePredict(trees[0],testdata_only[i])
            C1list.append(predictionsFiveVar[0])#C1list：n*40
            C2list.append(predictionsFiveVar[1])#C1list：n*40
            C3list.append(predictionsFiveVar[2])#C1list：n*40
            C4list.append(predictionsFiveVar[3])#C1list：n*40
            C5list.append(predictionsFiveVar[4])#C1list：n*40
        y_hat=[C1list,C2list,C3list,C4list,C5list]
        return y_hat

    '''Conducting the response predictions by means of the mlrp-rf models'''
    def perdictByforest(forest,testdata_only):
        """
        :param forest: The forest models in the form of dictionary
        :param testdata_only:  The testing samples
        :return: Return the response predictions of the five controlled variables
        """
        sum0=0
        sum1=0
        sum2=0
        sum3=0
        sum4=0
        for tree in forest:
            y_hat_fiveVar=predictTestdtaByOneTree(trees=tree,testdata_only=testdata_only)
            sum0=np.array(y_hat_fiveVar[0])+sum0#对应元素相加y_hat_fiveVa[0]:n*40  sum0:n*40
            sum1=np.array(y_hat_fiveVar[1])+sum1
            sum2=np.array(y_hat_fiveVar[2])+sum2
            sum3=np.array(y_hat_fiveVar[3])+sum3
            sum4=np.array(y_hat_fiveVar[4])+sum4
        sum0Average=sum0/len(forest)#sum0Average:n*40
        sum1Average=sum1/len(forest)
        sum2Average=sum2/len(forest)
        sum3Average=sum3/len(forest)
        sum4Average=sum4/len(forest)
        yhat_average=[sum0Average,sum1Average,sum2Average,sum3Average,sum4Average]#yhat_average：5*n*40
        return yhat_average
    yhatFiveVar=perdictByforest(forest=forest,testdata_only=test_data_only)
    return yhatFiveVar


if __name__ == '__main__':
    m1,m2,m3,m4,m5,m6,m7,m8=getControlID(startID=15905459,endID=15944125)
    manipulatedlist=[m1,m2,m3,m4,m5,m6,m7,m8]
    manipulatedlist2=[m1,m2,m3,m7]
    #Choosing the control antecedents correspond to the specific MLRP-RF model
    # controllist=getControlAntecedent1(controlSummaridlist=manipulatedlist2)
    # controllist=getControlAntecedent2(controlSummaridlist=manipulatedlist2)
    # controllist=getControlAntecedent3(controlSummaridlist=manipulatedlist2)
    # controllist=getControlAntecedent4(controlSummaridlist=manipulatedlist2)
    controllist=getControlAntecedent5(controlSummaridlist=manipulatedlist2)
    chronicle_list,tag_list,tag_change_list=Traningdata(controlidlist=controllist)#chroniclelist:n*2*10; tag_list:n*5*40; tag_change_list:n*5*40
    '''load the trained model and save the prediction results into the excel'''
    model = joblib.load(filename='modele2_C_2.pkl')
    yhatFiveVar=RFRpredicting(forest=model, test_data_only=chronicle_list)#yhatFiveVar：5*n*40
    tagfiveVar_origin = []
    for c in range(np.shape(tag_change_list)[1]):
        ctag_origin_list = []
        for n in range(len(tag_list)):
            ctag_origin_list.append(np.array(tag_list)[n][c, :])
        tagfiveVar_origin.append(ctag_origin_list)  # tagfiveVar:5*n*40
    hatfiveVar = []
    for c in range(np.shape(yhatFiveVar)[0]):
        c_hat_list = []
        for n in range(np.shape(yhatFiveVar)[1]):
            c_hat_list.append(np.array(yhatFiveVar[c][n, :]) / 100 + np.array(tagfiveVar_origin)[c][n, 0])  # c_hat_list:n*40
        hatfiveVar.append(c_hat_list)  # hatfiveVar:5*n*40
    tagfiveVar_origin_transform = np.reshape(tagfiveVar_origin, (5, -1)).transpose()
    hatfiveVar_trandorm = np.reshape(hatfiveVar, (5, -1)).transpose()
    #write to book
    workbook=xlsxwriter.Workbook("excel\\resultsModele2_C1_2.xlsx")
    worksheet=workbook.add_worksheet()
    for i in range(len(tagfiveVar_origin_transform)):
        for j in range(np.shape(tagfiveVar_origin_transform)[1]):
            worksheet.write(i,j,tagfiveVar_origin_transform[i,j])
            worksheet.write(i,j+6,hatfiveVar_trandorm[i,j])
    workbook.close()
