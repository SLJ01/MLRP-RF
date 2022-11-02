# -*- coding: utf-8 -*-
# @File   : MLRP_RF_training
# @Time   : 2022/11/2 15:17 
# @Author : linjin😀
import pymysql
import pandas as pd
import numpy as np
from numpy import inf
import xlsxwriter
from multiprocessing import Manager
import multiprocessing as mp
from sklearn.externals import joblib

controlnum=1
weidunumber=5
def test():
    suiji=np.random.RandomState()
    dimvar_index=[]
    indexlen=0
    while(indexlen<4):
        dimvar_index.append(suiji.randint(80))
        indexlen=len(set(dimvar_index))
    print('dimvar_index',dimvar_index)
    print('indexlen',indexlen)

'''Get the IDs of all manipulated variables when they changed  '''
def getControlID(startID,endID):
    """
    :param startID:iD for starting selecting samples
    :param endID:id for ending selecting samples
    :return:Returns the IDs of all manipulated variable changes
    """
    conn = pymysql.connect(host='10.11.112.202',port=3306,user='user',password='123456',db='SHUINI')
    sql = "select ID,TIMESTAMP,L0024,L0033,L0167,L0191,L0093,L0022,L0029,L0060,L0069 from DB_JIANGYIN where id  between %s and %s" %(startID,endID) # 15895459 and 16667985
    slj = pd.read_sql(sql, conn)
    slj = slj.sort_values(by='ID', axis=0, ascending=True)
    data_arr = np.array(slj)
    l0024_temp = 0
    l0033_temp = 0
    l0167_temp = 0
    l0191_temp=0
    l0093_temp=0
    l0022_temp=0
    l0029_temp=0
    l0060_temp=0
    idlist_l0024 = []
    idlist_l0033 = []
    idlist_l0167 = []
    idlist_l0191=[]
    idlist_l0093=[]
    idlist_l0022=[]
    idlist_l0029=[]
    idlist_l0060=[]
    for i in range(len(data_arr)):
        controlID = data_arr[i, 0]
        l0024 = data_arr[i, 2]
        l0033 = data_arr[i, 3]
        l0167 = data_arr[i, 4]
        l0191=data_arr[i,5]
        l0093=data_arr[i,6]
        l0022=data_arr[i,7]
        l0029=data_arr[i,8]
        l0060=data_arr[i,9]
        l0069=data_arr[i,-1]
        if (l0024 != l0024_temp):
            if(l0069>85):
                idlist_l0024.append(controlID)
            l0024_temp = l0024
        if (l0033 != l0033_temp):
            if (l0069 > 85):
                idlist_l0033.append(controlID)
            l0033_temp = l0033
        if (l0167 != l0167_temp):
            if (l0069 > 85):
                idlist_l0167.append(controlID)
            l0167_temp = l0167
        if(l0191!=l0191_temp):
            if(l0069>85):
                idlist_l0191.append(controlID)
            l0191_temp=l0191
        if(l0093!=l0093_temp):
            if(l0069>85):
                idlist_l0093.append(controlID)
            l0093_temp=l0093
        if(l0022!=l0022_temp):
            if(l0069>85):
                idlist_l0022.append(controlID)
            l0022_temp=l0022
        if(l0029!=l0029_temp):
            if(l0069>85):
                idlist_l0029.append(controlID)
            l0029_temp=l0029
        if(l0060!=l0060_temp):
            if(l0069>85):
                idlist_l0060.append(controlID)
            l0060_temp=l0060
    idlist_l0024 = idlist_l0024[1:]
    idlist_l0033 = idlist_l0033[1:]
    idlist_l0167 = idlist_l0167[1:]
    idlist_l0191 = idlist_l0191[1:]
    idlist_l0093=idlist_l0093[1:]
    idlist_l0022 = idlist_l0022[1:]
    idlist_l0029 = idlist_l0029[1:]
    idlist_l0060 = idlist_l0060[1:]
    return idlist_l0024, idlist_l0167,idlist_l0033,idlist_l0022,idlist_l0093,idlist_l0029,idlist_l0060,idlist_l0191
'''Judging that no other operations have occurred 
in the entire control strategy from idlist1 to idlist2'''
def ifNootheroperaton(idlist1,idlist2,t1,t2):
    """
    :param idlist1: the first candicate id list
    :param idlist2: the second candicate id list
    :param t1: the index in the first id list
    :param t2:the index in the second id list
    :return:Returns a sequence of IDs that only
    control operations on id1 and id2
    """
    if(idlist1[t1]<idlist2[t2]):
        if(t1==0 and t2!=len(idlist2)):
            if((idlist2[t2+1]-idlist2[t2])>30):
                return idlist1[0],idlist2[t2]
        if(t1!=0 and t2!=len(idlist2)):
            if((idlist1[t1]-idlist1[t1-1])>30 and (idlist2[t2+1]-idlist2[t2])>30):
                return idlist1[t1],idlist2[t2]
        if (t1!=0 and t2==len(idlist2)):
            if((idlist1[t1]-idlist1[t1-1])>30):
                return idlist1[t1],idlist2[len(idlist2)]
    else:
        if(t2==0 and t1!=len(idlist1)):
            if((idlist1[t1+1]-idlist1[t1])>30):
                return idlist2[0],idlist1[t1]
        if(t2!=0 and t1!= len(idlist1)):
            if ((idlist2[t2]-idlist2[t2-1])>30 and (idlist1[t1+1]-idlist1[t1])>30):
                return idlist2[t2],idlist1[t1]
        if(t2!=0 and t1==len(idlist1)):
            if((idlist2[t2]-idlist2[t2-1])>30):
                return idlist2[t2], idlist1[len(idlist1)]
        return None

'''Mining the first control antecednet from historical data for
 subsequent instance identification'''
def getControlAntecedent1(controlSummaridlist):
    """
    :param controlSummaridlist: A list of control ids for the given control strategy
    :return:Returns a list of ids for controlantecedent 1
    """
    m1_list=controlSummaridlist[0]#len:150
    m2_list=controlSummaridlist[1]#len:321
    m3_list=controlSummaridlist[2]#len204
    m7_list=controlSummaridlist[3]#len:39
    antecdedent1_list=[]
    for item in m1_list:
        m2_chazhi_=abs(np.array(m2_list)-item)
        m2_chazhi_paixu=sorted(m2_chazhi_)
        m1_chazhi=abs(np.array(m1_list)-item)
        m1_chazhi_paixu=sorted(m1_chazhi)
        m3_chazhi=abs(np.array(m3_list)-item)
        m3_chazhi_paixu=sorted(m3_chazhi)
        m7_chazhi=abs(np.array(m7_list)-item)
        m7_chazhi_paixu=sorted(m7_chazhi)
        if((m2_chazhi_paixu[0]<30) and m2_chazhi_paixu[1]>30 and (m1_chazhi_paixu[1]>30)):#说明30个点内只发生一次m1，也只发生一次m2
        # if((m2_chazhi_paixu[0]<30) and (m2_chazhi_paixu[1]>30) ):#说明30个点内只发生一次m1，也只发生一次m2
            if((m3_chazhi_paixu[0]>30) and (m7_chazhi_paixu[0]>30)):#说明30店内不发生m3和m7
                m2_chazhi_arr=np.array(m2_chazhi_)
                index_chazhi=np.where(m2_chazhi_arr==m2_chazhi_paixu[0])
                m2_id=np.array(m2_list)[index_chazhi].tolist()[0]
                m1_id=item
                id_couple=[m1_id,m2_id]
                antecdedent1_list.append(id_couple)
    return antecdedent1_list

'''Mining the second control antecednet from historical data for
 subsequent instance identification'''
def getControlAntecedent2(controlSummaridlist):
    """
    :param controlSummaridlist: A list of control ids for the given control strategy
    :return: Returns a list of ids for controlantecedent 2
    """
    m1_list=controlSummaridlist[0]#len:150
    m2_list=controlSummaridlist[1]#len:321
    m3_list=controlSummaridlist[2]#len204
    m7_list=controlSummaridlist[3]#len:39
    antecdedent2_list = []
    for item in m3_list:
        m3_cahzhi=abs(np.array(m3_list)-item)
        m3_chazhi_paixu=sorted(m3_cahzhi)
        m1_chazhi = abs(np.array(m1_list) - item)
        m1_chazhi_paixu = sorted(m1_chazhi)
        m2_chazhi = abs(np.array(m2_list) - item)
        m2_chazhi_paixu = sorted(m2_chazhi)
        m7_chazhi = abs(np.array(m7_list) - item)
        m7_chazhi_paixu = sorted(m7_chazhi)
        if((m3_chazhi_paixu[1]>30) and (m1_chazhi_paixu[0]>30) and (m2_chazhi_paixu[0]>30) and (m7_chazhi_paixu[0])>30):
            m1_id = item
            id_couple=[m1_id]
            antecdedent2_list.append(id_couple)
    return antecdedent2_list

'''Mining the third control antecednet from historical data for
 subsequent instance identification'''
def getControlAntecedent3(controlSummaridlist):
    """
    :param controlSummaridlist:  A list of control ids for the given control strategy
    :return: Returns a list of ids for controlantecedent 3
    """
    m1_list = controlSummaridlist[0]  # len:150
    m2_list = controlSummaridlist[1]  # len:321
    m3_list = controlSummaridlist[2]  # len204
    m7_list = controlSummaridlist[3]  # len:39
    antecdedent3_list = []
    for item in m7_list:
        m3_cahzhi = abs(np.array(m3_list) - item)
        m3_chazhi_paixu = sorted(m3_cahzhi)
        m1_chazhi = abs(np.array(m1_list) - item)
        m1_chazhi_paixu = sorted(m1_chazhi)
        m2_chazhi = abs(np.array(m2_list) - item)
        m2_chazhi_paixu = sorted(m2_chazhi)
        m7_chazhi = abs(np.array(m7_list) - item)
        m7_chazhi_paixu = sorted(m7_chazhi)
        if ((m3_chazhi_paixu[0] > 30) and (m1_chazhi_paixu[0] > 30) and (m2_chazhi_paixu[0] > 30) and (
        m7_chazhi_paixu[1]) > 30):
            m1_id = item
            id_couple = [m1_id]
            antecdedent3_list.append(id_couple)
    return antecdedent3_list

'''Mining the third control antecednet from historical data for
 subsequent instance identification'''
def getControlAntecedent4(controlSummaridlist):
    """
    :param controlSummaridlist:  A list of control ids for the
    given control strategy
    :return:  Returns a list of ids for controlantecedent 4
    """
    m1_list = controlSummaridlist[0]  # len:150
    m2_list = controlSummaridlist[1]  # len:321
    m3_list = controlSummaridlist[2]  # len204
    m7_list = controlSummaridlist[3]  # len:39
    antecdedent4_list = []
    for item in m1_list:
        m3_cahzhi = abs(np.array(m3_list) - item)
        m3_chazhi_paixu = sorted(m3_cahzhi)
        m1_chazhi = abs(np.array(m1_list) - item)
        m1_chazhi_paixu = sorted(m1_chazhi)
        m2_chazhi = abs(np.array(m2_list) - item)
        m2_chazhi_paixu = sorted(m2_chazhi)
        m7_chazhi = abs(np.array(m7_list) - item)
        m7_chazhi_paixu = sorted(m7_chazhi)
        if ((m3_chazhi_paixu[0] < 30) and (m1_chazhi_paixu[1] > 30) and (m2_chazhi_paixu[0] < 30) and (
                m7_chazhi_paixu[0]) < 30):
            m1_id=item
            id_couple = [m1_id]
            antecdedent4_list.append(id_couple)
    return antecdedent4_list

'''Mining the third control antecednet from historical data for
 subsequent instance identification'''
def getControlAntecedent5(controlSummaridlist):
    """
    :param controlSummaridlist:  A list of control ids for the
    given control strategy
    :return:  Returns a list of ids for controlantecedent 5
    """
    m1_list = controlSummaridlist[0]  # len:150
    m2_list = controlSummaridlist[1]  # len:321
    m3_list = controlSummaridlist[2]  # len204
    m7_list = controlSummaridlist[3]  # len:39
    antecdedent5_list = []
    for item in m2_list:
        m3_cahzhi = abs(np.array(m3_list) - item)
        m3_chazhi_paixu = sorted(m3_cahzhi)
        m1_chazhi = abs(np.array(m1_list) - item)
        m1_chazhi_paixu = sorted(m1_chazhi)
        m2_chazhi = abs(np.array(m2_list) - item)
        m2_chazhi_paixu = sorted(m2_chazhi)
        m7_chazhi = abs(np.array(m7_list) - item)
        m7_chazhi_paixu = sorted(m7_chazhi)
        if((m3_chazhi_paixu[0] > 30) and (m1_chazhi_paixu[0] > 30) and (m7_chazhi_paixu[0] > 30) and (m2_chazhi_paixu[1]) > 30):
            m1_id=item
            id_couple=[item]
            antecdedent5_list.append(id_couple)
    return antecdedent5_list

'''Tool: Save mined chronicle data to excel'''
def xieruExcel(chronicledata) -> object:
    """
    :param chronicledata:
    :return:Returns the excel save path
    """
    #chronicledata is two-dimension
    url="excel\\trainingdataForModel5.xlsx"#
    workbook=xlsxwriter.Workbook(url)
    worksheet=workbook.add_worksheet()
    for i in range(len(chronicledata)):
        for j in range(np.shape(chronicledata)[1]):
            worksheet.write(i,j,chronicledata[i,j])
    workbook.close()
    return url

'''Obtain training samples and their corresponding tags
 according to the ID of the control occurrence'''
def Traningdata(controlidlist):
    """
    :param controlidlist: A list of Ids when the control occurs
    :return:Return training samples and their tags
    """
    chronicledata_list=[]
    tag_list=[]
    tag_change_list=[]
    for idcouple in controlidlist:
        conn=pymysql.connect(host='10.11.112.202', port=3306,user='user',password='123456',db='SHUINI')
        sql="select ID,TIMESTAMP,L0024,L0167,L0033,L0060,L0022,L0093,L0029,L0191,L0069,L0026,L0027,L0050,L0070," \
            "L0034,L0039,L0040,L0041,L0042,L0049,L0061,L0070,L0094,L0095,L0097,L0157,L0158,L0159,L0162,L0013,L0163,L0164,L0002,L0005,L0008 from DB_JIANGYIN where id between %s and %s" %(idcouple[0]-30,idcouple[0]+80)#Take out the 30 points before the control to calculate the slope
        slj=pd.read_sql(sql,conn)
        slj=slj.sort_values(by='ID',axis=0,ascending=True)
        data_arr=np.array(slj)
        #Extract tags of all control variables
        tag_C1=data_arr[30:70,10]#Vertical mill outlet temperature L0069
        tag_C2=data_arr[30:70,11]#Furnace temperature L0026
        tag_C3=data_arr[30:70,13]#Vertical grinding vibration L0050
        tag_C4=data_arr[30:70,12]#Furnace pressure L0027
        tag_C5=data_arr[30:70,14]#Vertical mill outlet pressure L0070
        init_C1=tag_C1[0]
        init_C2=tag_C2[0]
        init_C3=tag_C3[0]
        init_C4=tag_C4[0]
        init_C5=tag_C5[0]
        changetag_C1=(tag_C1-init_C1)*100
        changetag_C2=(tag_C2-init_C2)*100
        changetag_C3=(tag_C3-init_C3)*100
        changetag_C4=(tag_C4-init_C4)*100
        changetag_C5=(tag_C5-init_C5)*100
        tag=[tag_C1,tag_C2,tag_C3,tag_C4,tag_C5]
        tag_change=[changetag_C1,changetag_C2,changetag_C3,changetag_C4,changetag_C5]

        '''MIning chronicle data to prepare for subsequent model training'''
        def preparedChronicledata(data_arr,controlnum=controlnum):
            """
            :param data_arr: The origial control process data
            :param controlnum: The number of control operations
            in a given control strategy
            :return: return the control sample data
            """
            feed=np.c_[data_arr,(data_arr[:,-1]+data_arr[:,-2]+data_arr[:,-3])]
            plusFeed=np.delete(feed,(33,34,35),axis=1)
            # init_manipulated=[data_arr[1,2],data_arr[1,3],data_arr[1,4],data_arr[1,5],data_arr[1,6],data_arr[1,7],data_arr[1,8],data_arr[1,9]]#m1,m2,m3,m4,m5,m6,m7.m8
            init_manipulated=[data_arr[1,2],data_arr[1,3],data_arr[1,4],data_arr[1,5]]#m1,m2,m3,m7
            #Initialize the chronicle dimension variable
            changeid=[]
            changeManipulated=[]
            changeParameter=[]
            changetime=[]
            antecednetC1=[]
            antecednetC2=[]
            antecednetC3=[]
            antecednetC4=[]
            antecednetC5=[]
            for m in range(2,6,1):#m=2,3,4,5
                for i in range(len(plusFeed)):
                    if(plusFeed[i,m]!=init_manipulated[m-2]):#Indicates that a control operation has occurred
                        changeid.append(plusFeed[i,0])
                        changeManipulated.append(plusFeed[i,m]-init_manipulated[m-2])#save the changes of manipulated variables
                        changeParameter.append(m-1)#0：m1,1:m2 2:m3...
                        if(i<5):
                            startindex=0
                        else:
                            startindex=i-5
                        antecednetC1.append(np.average(list(plusFeed[startindex:i+1,10])))
                        antecednetC2.append(np.average(list(plusFeed[startindex:i+1,11])))
                        antecednetC3.append(np.average(list(plusFeed[startindex:i+1,13])))
                        antecednetC4.append(np.average(list(plusFeed[startindex:i+1,12])))
                        antecednetC5.append(np.average(list(plusFeed[startindex:i+1,14])))
                        changetime.append(plusFeed[i,1])
                        init_manipulated[m-2]=plusFeed[i,m]
            changeTime_sort = np.array(changetime)[np.argsort(np.array(changetime))]  # Sort the time first, and then subtract
            # the previous one to get the event time interval that occurred.
            timeGap = [float(changeTime_sort[i]) - float(changeTime_sort[i - 1]) for i in range(1, len(changetime))]
            timeGap.insert(0, 0)  # Add 0 to the timeGap header
            getchange=np.array([list(zip(changeid,changetime,changeManipulated,changeParameter,antecednetC1,antecednetC2,antecednetC3,antecednetC4,antecednetC5))])[0]
            getchange=getchange[np.argsort(getchange[:,0]),:]#Sort by id from smallest to largest
            timeGap=np.array(timeGap)/1000#Convert milliseconds to seconds
            getchange = np.insert(getchange, 4, timeGap, axis=1)
            return getchange[:controlnum]
        chronicledata=preparedChronicledata(data_arr[:])
        chronicledata_list.append(chronicledata)
        tag_list.append(tag)
        tag_change_list.append(tag_change)
    '''The following two variables are converted into 
    two-dimensional arrays and stored in excel'''
    # #chronicledata_list：n*2*10;tag_list:n*5*10
    # chronicledata_arr=np.array(chronicledata_list)
    # chronicledata_arr_twodimen=np.reshape(chronicledata_arr,(-1,10))
    # tag_arr=np.array(tag_list)
    # tag_arr_twodimen=np.reshape(tag_arr,(-1,40))
    #
    # tag_change_arr=np.array(tag_change_list)
    # tag_change_arr_twodimen=np.reshape(tag_change_arr,(-1,40))
    #
    #
    # #save unlabeled into excel
    # url_tarining=xieruExcel(chronicledata_arr_twodimen)
    # #save labels to excel
    # # url_tag=xieruExcel(tag_change_arr_twodimen)
    # breakpoint()
    return chronicledata_list,tag_list,tag_change_list


'''The main training process with the given chronicle data'''
def MLRP_TRAINING(chronicledata_list,tag_list,tag_change_list,treeNum,suiji,return_dict,processnum):
    """
    :param chronicledata_list:The prepared training data samples
    :param tag_list: The prepared tag list
    :param tag_change_list: The perpared change value of tag list
    :param treeNum:The number of trees in each cascade in each base learners
    :param suiji: The random seed
    :param return_dict: Save the forest models in dictionary form
    :param processnum: Number of processes using parallel computing
    :return:Returns the model with the tailored MLRP-RF saved
    """
    '''compulate the slope based on the value of given variables '''
    def getSlope(onecontrolVartag,prexnum):
        """
        :param onecontrolVartag: The tag value of one given controlled variables
        :param prexnum:  The number of tags used to calculate the slope
        :return: Returns the calculated slope value for a given number of tags
        """
        slope_onecontrolVartag=[]
        for i in range(prexnum-1,len(onecontrolVartag)):
            prexTag=onecontrolVartag[i-prexnum+1:i+1]
            z_axis=np.arange(1,len(prexTag)+1,1)
            z1=np.polyfit(z_axis,prexTag,1)
            p2=np.poly1d(z1)
            ope=p2.coef[0]
            slope_onecontrolVartag.append(ope)
        return slope_onecontrolVartag

    '''Divide the sampledata according to the given segmentation variable 
    and segmentation threshold'''
    def Splitedata(sampledata,tag_change,tag_data,fengeVar,fengeValue):
        """
        :param sampledata: The samples to be splited
        :param tag_change: The change values of tag
        :param tag_data: The tag of the given samples
        :param fengeVar: The candicate vaiables that would be chosen to splite the sample data
        :param fengeValue: The candicate thresholds of cancidate variables chosen to splite the sample data
        :return:Returns the original value, the label value and the changed value of the label divided into left and right samples
        """
        sampledata_zip=np.zeros((np.shape(sampledata)[0],np.shape(sampledata)[2]))
        #Compress the original sample data and then split it, because the value I split is obtained from the compressed sample_zip
        for i in range(np.shape(sampledata)[0]):
            for j in range(np.shape(sampledata)[2]):
                sampledata_zip[i,j]=np.mean(sampledata[i][:,j].astype('float'))
        leftindex=np.where(sampledata_zip[:,fengeVar]>fengeValue)[0]
        leftsamples=np.array(sampledata)[leftindex,:]#The leftindex row of all columns is taken out
        rightindex=np.where(sampledata_zip[:,fengeVar]<=fengeValue)[0]
        rightsamples=np.array(sampledata)[rightindex,:]
        left_tag=np.array(tag_data)[leftindex,:]
        left_changetag=np.array(tag_change)[leftindex,:]
        right_tag=np.array(tag_data)[rightindex,:]
        righ_changetag=np.array(tag_change)[rightindex,:]
        return leftsamples,left_tag,left_changetag,rightsamples,right_tag,righ_changetag

    '''Calculate the sum of impurities of all controlled Variables based on the inpurity function'''
    def computeImpuritAfterSplit_111(left_changetag,right_changetag):
        """
        :param left_changetag: The change value of the left tag
        :param right_changetag: The change value of the right tag
        :return: Return the impurity of the left and right samples
        """
        #left_changetag:n*5*40
        left_changetag=np.array(left_changetag)
        right_changetag=np.array(right_changetag)
        if(len(left_changetag)==0):
            left_impurity=0
        else:
            impurity_sumvar_left=np.zeros(np.shape(left_changetag)[1])
            for c in range(np.shape(left_changetag)[1]):
                changetag_ConrolC_list=[]
                for n in range(len(left_changetag)):
                   changetag_ConrolC_list.append(left_changetag[n][c,:])#changetag_ConrolC_list:n*40
                averge_ControlC=np.average(changetag_ConrolC_list,axis=0)#averge_ControlC:1*40
                average_ControlC_slope=getSlope(onecontrolVartag=averge_ControlC,prexnum=4)
                # Pure numerical impurity
                left_valueimpurity=np.sum(abs(np.array(averge_ControlC)-np.array(changetag_ConrolC_list)),axis=1)#1*n impurity, n represents the number of samples
                #Slope impurity is considere
                ControlC_slope=[]
                for i in range(len(changetag_ConrolC_list)):
                    ControlC_slope.append(getSlope(onecontrolVartag=changetag_ConrolC_list[i],prexnum=4))#ControlC_slope：n*(40-4)
                left_slopeimpurity=np.sum(abs(np.array(ControlC_slope)-np.array(average_ControlC_slope)),axis=1)
                #The value and the slope z-score are normalized and then accumulated
                std_value=np.std(left_valueimpurity)
                std_slope=np.std(left_slopeimpurity)
                if(std_value==0):
                    std_value=0.01
                if(std_slope==0):
                    std_slope=0.01
                a = (left_valueimpurity - np.mean(left_valueimpurity)) /std_value
                b = (left_slopeimpurity - np.mean(left_slopeimpurity)) /std_slope
                impurity_left = np.sum(a + b)  #
                impurity_sumvar_left[c]=impurity_left
            left_impurity=impurity_sumvar_left
        left_sumImpurity=np.sum(left_impurity)#The sum of the impurity of all control variables in the left sample
        if(len(right_changetag)==0):
            right_impurity=0
        else:
            impurity_sumvar_right=np.zeros(np.shape(right_changetag)[1])
            for c in range(np.shape(right_changetag)[1]):
                changetag_ConrolC_list2=[]
                for n in range(len(right_changetag)):
                    changetag_ConrolC_list2.append(right_changetag[n][c,:])
                average_ControlC2=np.average(changetag_ConrolC_list2,axis=0)
                average_ControlC2_slope=getSlope(onecontrolVartag=average_ControlC2,prexnum=4)
                right_valueimpurity=np.sum(abs(np.array(average_ControlC2)-np.array(changetag_ConrolC_list2)),axis=1)
                ControlC_slope2=[]
                for i in range(len(changetag_ConrolC_list2)):
                    ControlC_slope2.append(getSlope(onecontrolVartag=changetag_ConrolC_list2[i],prexnum=4))
                right_slopeimpurity=np.sum(abs(np.array(ControlC_slope2)-np.array(average_ControlC2_slope)),axis=1)
                std_value2=np.std(right_valueimpurity)
                std_slope2=np.std(right_slopeimpurity)
                if(std_value2==0):
                    std_value2=0.01
                if(std_slope2==0):
                    std_slope2=0.01
                a = (right_valueimpurity - np.mean(right_valueimpurity)) / std_value2
                b = (right_slopeimpurity - np.mean(right_slopeimpurity)) / std_slope2
                impurity_right = np.sum(a + b)
                # impurity_right=np.sum(right_valueimpurity/100+right_slopeimpurity)
                impurity_sumvar_right[c]=impurity_right
            right_impurity=impurity_sumvar_right
        right_sumImpurity=np.sum(right_impurity)
        impurity=(right_sumImpurity+left_sumImpurity)/(len(left_changetag)+len(right_changetag))
        return impurity

    '''Calculate the sum of the impurity of the left and right samples under the given segmentation variable'''
    def computeImpurityAfterSplit_VarC(left_changetag, right_changetag, c):  # c=0,1,2,3,4-->c1,c2,c3,c4,c5
        """
        :param left_changetag:The change value of the left tag
        :param right_changetag:The change value of the right tag
        :param c: The index of controlled variables
        :return:Return the impurity of the left and right samples
        """
        left_changetag=np.array(left_changetag)
        right_changetag=np.array(right_changetag)
        if (len(left_changetag) == 0):
            left_impurity = 0
        else:
            changetag_C_list=[]
            for n in range(len(left_changetag)):
                changetag_C_list.append(left_changetag[n][c,:])
            averge_ControlC=np.average(changetag_C_list,axis=0)
            averge_ControlC_slope=getSlope(onecontrolVartag=averge_ControlC,prexnum=4)
            left_valueimpurity=np.sum(abs(np.array(changetag_C_list)-np.array(averge_ControlC)),axis=1)
            ControlC_slope=[]
            for i in range(len(changetag_C_list)):
                ControlC_slope.append(getSlope(onecontrolVartag=changetag_C_list[i],prexnum=4))
            left_slopeimpurity=np.sum(abs(np.array(ControlC_slope)-np.array(averge_ControlC_slope)),axis=1)
            std_value = np.std(left_valueimpurity)
            std_slope = np.std(left_slopeimpurity)
            if (std_value == 0):
                std_value = 0.01
            if (std_slope == 0):
                std_slope = 0.01
            a=(left_valueimpurity-np.mean(left_valueimpurity))/std_value
            b=(left_slopeimpurity-np.mean(left_slopeimpurity))/std_slope
            left_impurity=np.sum(a+b)
        if (len(right_changetag) == 0):
            right_impurity = 0
        else:
            changetag_C_list2=[]
            for n in range(len(right_changetag)):
                changetag_C_list2.append(right_changetag[n][c,:])
            averge_ControlC2=np.average(changetag_C_list2,axis=0)
            averge_ControlC_slope2=getSlope(onecontrolVartag=averge_ControlC2,prexnum=4)
            right_valueimpurity=np.sum(abs(np.array(changetag_C_list2)-np.array(averge_ControlC2)),axis=1)
            ControlC_slope2=[]
            for i in range(len(changetag_C_list2)):
                ControlC_slope2.append(getSlope(onecontrolVartag=changetag_C_list2[i],prexnum=4))
            right_slopeimpurity=np.sum(abs(np.array(ControlC_slope2)-np.array(averge_ControlC_slope2)),axis=1)
            std_value2 = np.std(right_valueimpurity)
            std_slope2 = np.std(right_slopeimpurity)
            if (std_value2 == 0):
                std_value2 = 0.01
            if (std_slope2 == 0):
                std_slope2 = 0.01
            a=(right_valueimpurity-np.mean(right_valueimpurity))/std_value2
            b=(right_slopeimpurity-np.mean(right_slopeimpurity))/std_slope2
            right_impurity=np.sum(a+b)
        impurity = (right_impurity + left_impurity) / (len(left_changetag) + len(right_changetag))
        return impurity

    '''Returns the best segmentation Index, the best segmentation value, 
    the number of left samples, the number of right samples'''
    def getBestfeatureAndValue(smapledata,tag_data,tag_change,m,suiji):
        """
        :param smapledata: The training samples
        :param tag_data:  The sample tag
        :param tag_change: The change value of tag data
        :param m:m represents the number of dimension variables to be selected for one split
        :param suiji: The random seed
        :return:Returns the best segmentation Index, the best segmentation value,
                the number of left samples, the number of right samples
        """
        dimension=np.shape(smapledata)[2]
        dimIndex=[]
        count=0
        while(count<m):
            index=suiji.randint(dimension)
            if(index==2 or index==4 or index==5 or index==6 or index==7 or index==8 or index==9):
                dimIndex.append(index)
                count=len(set(dimIndex))
        dimIndex=set(dimIndex)#remove duplicate elements
        if (len(smapledata)<=1):#return the average values of tags
            leaf_averagelist=[]
            for c in range(np.shape(np.array(tag_change))[1]):
                c1tag_list=[]
                for i in range(len(tag_change)):
                    c1tag_list.append(tag_change[i][c])
                c1tag_average=np.average(c1tag_list,axis=0)
                leaf_averagelist.append(c1tag_average)
            return None,leaf_averagelist
        sampledata_zip=np.zeros((np.shape(smapledata)[0],np.shape(smapledata)[2]))
        for i in range(len(smapledata)):
            for j in range(np.shape(smapledata)[2]):
                sampledata_zip[i,j]=np.mean(smapledata[i][:,j].astype('float'))
        #Start spliting....
        best_dindex_list=[]
        best_value_list=[]
        best_leftsamples=[]
        best_rightsamples=[]
        for dindex in dimIndex:
            bestPurity = inf
            for value in set(sampledata_zip[:,dindex]):
                leftsample,left_tag,left_changetag,rightsample,right_tag,rightchangetag=Splitedata(sampledata=smapledata,tag_change=tag_change,tag_data=tag_data,fengeVar=dindex,fengeValue=value)
                newPurity=computeImpurityAfterSplit_VarC(left_changetag=left_changetag,right_changetag=rightchangetag,c=1)
                if(newPurity<bestPurity):
                    best_dindex_list.append(dindex)
                    best_value_list.append(value)
                    bestPurity=newPurity
                    best_leftsamples.append(leftsample)
                    best_rightsamples.append(rightsample)
        if(len(best_dindex_list)==0):
            leaf_averagelist = []
            for c in range(np.shape(np.array(tag_change))[1]):
                c1tag_list = []
                for i in range(len(tag_change)):
                    c1tag_list.append(tag_change[i][c])
                c1tag_average = np.average(c1tag_list, axis=0)
                leaf_averagelist.append(c1tag_average)
            return None, leaf_averagelist
        return best_dindex_list[-1],best_value_list[-1],len(best_leftsamples[-1]),len(best_rightsamples[-1])

    '''Develop  GBDTs ....'''
    def createTree(sampledata,tagdata,tag_change,maxdepth,suiji):
        """
        :param sampledata: The training data used to train trees
        :param tagdata: Tags for the sampledata
        :param tag_change: The change value of tags
        :param maxdepth: The maxdepth for each tree when training the forest models
        :param suiji: The random seeds
        :return: Returns the tree model represented as a dictionary
        """
        if (len(sampledata)<3):
            leaf_averagelist=[]
            for c in range(np.shape(np.array(tag_change))[1]):
                c1tag_list=[]
                for i in range(len(tag_change)):
                    c1tag_list.append(tag_change[i][c])
                c1tag_average=np.average(c1tag_list,axis=0)
                leaf_averagelist.append(c1tag_average)
            return leaf_averagelist
        best_dindex, best_value,leftsample_len,rightsample_len=getBestfeatureAndValue(smapledata=sampledata, tag_data=tagdata,tag_change=tag_change, m=weidunumber,suiji=suiji)
        count=0
        while (((leftsample_len<1) or(rightsample_len<1) )and (count<5)):
            best_dindex, best_value, leftsample_len, rightsample_len = getBestfeatureAndValue(smapledata=sampledata,tag_data=tagdata,tag_change=tag_change,m=weidunumber, suiji=suiji)
            count=count+1
        if(best_dindex==None):
            print('Get one tree model，the depth of the tree is :',20-maxdepth)
        Tree={}
        maxdepth=maxdepth-1
        if(maxdepth<0):
            print('Reach the deepest level of the tree')
            leaf_averagelist = []
            for c in range(np.shape(np.array(tag_change))[1]):
                c1tag_list = []
                for i in range(len(tag_change)):
                    c1tag_list.append(tag_change[i][c])
                c1tag_average = np.average(c1tag_list, axis=0)
                leaf_averagelist.append(c1tag_average)
            return leaf_averagelist
        Tree['best_dindex']=best_dindex
        Tree['best_value']=best_value
        leftsample,left_tag,left_changetag,rightsample,right_tag,rightchangetag=Splitedata(sampledata=sampledata,tag_change=tag_change,tag_data=tagdata,fengeVar=best_dindex,fengeValue=best_value)
        if((len(leftsample)<1) or (len(rightsample)<1)):
            leaf_averagelist = []
            for c in range(np.shape(np.array(tag_change))[1]):
                c1tag_list = []
                for i in range(len(tag_change)):
                    c1tag_list.append(tag_change[i][c])
                c1tag_average = np.average(c1tag_list, axis=0)
                leaf_averagelist.append(c1tag_average)
            return leaf_averagelist
        Tree['left']=createTree(sampledata=leftsample,tagdata=left_tag,tag_change=left_changetag,maxdepth=maxdepth,suiji=np.random.RandomState())
        Tree['right']=createTree(sampledata=rightsample,tagdata=right_tag,tag_change=rightchangetag,maxdepth=maxdepth,suiji=np.random.RandomState())
        return Tree
    treelist=[]
    for i in range(treeNum):
        oneTreee=createTree(sampledata=chronicledata_list,tagdata=tag_list,tag_change=tag_change_list,maxdepth=90,suiji=suiji)
        print('--------------Finish the ',i,'tree model--------------------')
        treelist.append(oneTreee)
    return_dict[processnum]=treelist

if __name__ == '__main__':
    m1,m2,m3,m4,m5,m6,m7,m8=getControlID(startID=15905459,endID=15944125)
    #getControlAntecedent1:(startID=27055148,endID=27135148);;;getControlAntecedent2:(startID=27055148,endID=27081814);;
    # getControlAntecedent4(startID=27055148,endID=27073814); getControlAntecedent5:(startID=15895459,endID=15934125)
    manipulatedlist=[m1,m2,m3,m4,m5,m6,m7,m8]
    manipulatedlist2=[m1,m2,m3,m7]#m2=m8
    '''Choosing the control antecedents correspond to the specific MLRP-RF model'''
    # controllist=getControlAntecedent1(controlSummaridlist=manipulatedlist2)
    # controllist=getControlAntecedent2(controlSummaridlist=manipulatedlist2)
    # controllist=getControlAntecedent3(controlSummaridlist=manipulatedlist2)
    # controllist=getControlAntecedent4(controlSummaridlist=manipulatedlist2)
    controllist=getControlAntecedent5(controlSummaridlist=manipulatedlist2)
    chronicle_list,tag_list,tag_change_list=Traningdata(controlidlist=controllist)#chroniclelist:n*2*10; tag_list:n*5*40; tag_change_list:n*5*40
    '''Training of MLRP-RF models parallelly'''
    manger = Manager()
    return_dict = manger.dict()
    process_list = []
    k=1
    for i in range(k):
        p = mp.Process(target=MLRP_TRAINING,
                       args=(chronicle_list, tag_list,tag_change_list, 4, np.random.RandomState(),return_dict,i))
        process_list.append(p)
        p.start()
    for item in process_list:
        item.join()
    forest_list = return_dict.values()
    joblib.dump(forest_list, filename='modele2_C_2.pkl', compress=3)#compress refers to the compression ratio




