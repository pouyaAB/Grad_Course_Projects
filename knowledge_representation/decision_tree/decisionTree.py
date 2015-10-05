# This code will work with this dataset
# https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
# It will create decision tree and classify the data in the dataset
# Assignment 1 for knowledge represetation course
import csv
from numpy import genfromtxt
import numpy as np
import math


class attribute:
    def __init__(self, name, index, type, options):
        self.name=name
        self.index=index
        self.type=type
        self.options=options
    def get_entropy(self, data):
        result=data[:,13]
        if len(result)>0:
            class_positive=result[np.where(result[:]>0)]
            class_negative=result[np.where(result[:]<0)]
            portion_positive=len(class_positive)/float(len(result))
            portion_negative=len(class_negative)/float(len(result))
            ##print "portion_positive, portion_negative "+ str(portion_positive)+"  "+str(portion_negative)
            right=0
            left=0
            if portion_positive>0:
                right=portion_positive*math.log(portion_positive,2)
            if portion_negative>0:
                left=portion_negative*math.log(portion_negative,2)
            return -(right+left)
        else:
            return 0
    def get_splited_data(self, data):
        toRet=[]
        feature_data=data[:,self.index]
        if self.type=='Real':
            in_option=data[np.where(feature_data[:]>=self.split_point)]
            out_option=data[np.where(feature_data[:]<self.split_point)]
            toRet.append(in_option)
            toRet.append(out_option)
        else:
            for option in self.options:
                in_option=data[np.where(feature_data[:]==option)]
                toRet.append(in_option)

        return toRet

    def gain(self, data):
        feature_data=data[:,self.index]
        result=data[:,13]

        parent_entropy=self.get_entropy(data)
        childs_entropy=0

        if self.type=='Real':
            self.split_point=self.findBestSplitPoint(data)

            in_option=feature_data[np.where(feature_data[:]>=self.split_point)]
            in_option_result=result[np.where(feature_data[:]>=self.split_point)]
            if len(in_option_result)>0:
                class_positive=in_option_result[np.where(in_option_result[:]>0)]
                class_negative=in_option_result[np.where(in_option_result[:]<0)]
                portion_positive=len(class_positive)/float(len(in_option_result))
                portion_negative=len(class_negative)/float(len(in_option_result))
                right=0
                left=0
                if portion_positive>0:
                    right=portion_positive*math.log(portion_positive,2)
                if portion_negative>0:
                    left=portion_negative*math.log(portion_negative,2)
                childs_entropy -=((right+left)*len(in_option_result)/len(result))

            in_option=feature_data[np.where(feature_data[:]<self.split_point)]
            in_option_result=result[np.where(feature_data[:]<self.split_point)]
            if len(in_option_result)>0:
                class_positive=in_option_result[np.where(in_option_result[:]>0)]
                class_negative=in_option_result[np.where(in_option_result[:]<0)]
                portion_positive=len(class_positive)/float(len(in_option_result))
                portion_negative=len(class_negative)/float(len(in_option_result))
                right=0
                left=0
                if portion_positive>0:
                    right=portion_positive*math.log(portion_positive,2)
                if portion_negative>0:
                    left=portion_negative*math.log(portion_negative,2)
                childs_entropy-=((right+left)*len(in_option_result)/len(result))

        else:
            for option in self.options:
                in_option=feature_data[np.where(feature_data[:]==option)]
                in_option_result=result[np.where(feature_data[:]==option)]
                if len(in_option_result)>0:
                    class_positive=in_option_result[np.where(in_option_result[:]>0)]
                    class_negative=in_option_result[np.where(in_option_result[:]<0)]
                    portion_positive=len(class_positive)/float(len(in_option_result))
                    portion_negative=len(class_negative)/float(len(in_option_result))
                    right=0
                    left=0
                    if portion_positive>0:
                        right=portion_positive*math.log(portion_positive,2)
                    if portion_negative>0:
                        left=portion_negative*math.log(portion_negative,2)
                    childs_entropy-=((right+left)*len(in_option_result)/len(result))

        return parent_entropy - childs_entropy

    def findBestSplitPoint(self, data):
        x=data[:,self.index]
        y=data[:,13]

        a1=np.sum(np.power(x,2))
        d1=np.sum(x)
        c1=np.sum(x*y)

        a2=np.sum(x)
        d2=len(x)
        c2=np.sum(y)

        a = np.array([[a1,a2], [d1,d2]])
        c = np.array([c1,c2])
        try:
            [w, b] = np.linalg.solve(a, c)
        except np.linalg.linalg.LinAlgError as err:
            return (max(data[:,self.index])+min(data[:,self.index]))/2
        ##print "w,b:"+str(w)+"  "+str(b)
        return -b/w


class decision_node:
    def __init__(self):
        #self.attribute=attribute
        self.child_nodes=[]
    def set_attribute(self, attribute):
        self.attribute=attribute
    def add_child(self, node_toAdd):
         self.child_nodes.append(node_toAdd)
    def set_tag(self,tag):
        self.tag=tag
    def predict(self,toTest):
        if(len(self.child_nodes)==0):
            return self.tag

        corresponding_var=toTest[self.attribute.index]
        if self.attribute.type=='Real':
            if(corresponding_var>=self.attribute.split_point):
                return self.child_nodes[0].predict(toTest)
            else:
                return self.child_nodes[1].predict(toTest)
        else:
            for i in range(len(self.child_nodes)):
                if corresponding_var==self.attribute.options[i]:
                    return self.child_nodes[i].predict(toTest)


def kfold(data, k):
    fold_size=math.floor(len(data)/k);
    return fold_size;


def train(data, attributes):
    #raw_input('Enter your input:')
    threshold=0.5
    root= decision_node()
    if attributes[0].get_entropy(data)>threshold:
        gains=[]
        for attr in attributes:
            #print attr.gain(data)
            gains.append(attr.gain(data))
        root_index=np.argmax(gains)
        root.set_attribute(attributes[root_index])
        #print root.attribute.name+" "+str(root.attribute.get_entropy(data))
        splited_data=root.attribute.get_splited_data(data)
        #print "child number: "+ str(len(splited_data))
        for branch_data in splited_data:
            #print "child size: "+ str(len(branch_data))
            #print branch_data
            toAdd_decision_node=train(branch_data,attributes)
            root.add_child(toAdd_decision_node)
    else:
        result=data[:,13]
        class_positive=result[np.where(result[:]>0)]
        class_negative=result[np.where(result[:]<0)]
        if(len(class_positive)>=len(class_negative)):
            root.set_tag(1);
            #print "POSITIVE"
        else:
            root.set_tag(-1);
            #print "NEGATIVE"
        #print ">>>>>>>>>>>>>>"
    return root

def main():
    data = genfromtxt('heart.dataset', delimiter=' ')
    data[:,13]=data[:,13]*2-3
    #data_t=np.transpose(data)
    #print data_t[13]

    attributes= []
    # Real: 1,4,5,8,10,12
    # Ordered:11,
    # Binary: 2,6,9
    # Nominal:7,3,13
    age= attribute('age', 0, 'Real', [29,77])
    sex= attribute('sex', 1, 'Binary', [0,1])
    pain_type= attribute('pain_type', 2, 'Nominal', [1.0,2.0,3.0,4.0])
    blood_pressure= attribute('blood_pressure', 3, 'Real', [94,200])
    cholestoral= attribute('cholestoral', 4, 'Real', [126, 564])
    blood_suger= attribute('blood_suger', 5, 'Binary', [0,1])
    electrocardiographic= attribute('electrocardiographic',6, 'Nominal', [0, 1, 2])
    max_heart_rate= attribute('max_heart_rate', 7, 'Real', [71, 202])
    exercise= attribute('exercise', 8, 'Binary', [0, 1])
    oldpeak= attribute('oldpeak', 9, 'Real', [0.0, 6.2])
    slope= attribute('slope', 10, 'Nominal', [1.0, 2.0, 3.0])
    vessels= attribute('vessels', 11, 'Nominal', [0.0, 1.0, 2.0, 3.0])
    thal= attribute('thal', 12, 'Nominal', [3, 6, 7])

    attributes.append(age)
    attributes.append(sex)
    attributes.append(pain_type)
    attributes.append(blood_pressure)
    attributes.append(cholestoral)
    attributes.append(blood_suger)
    attributes.append(electrocardiographic)
    attributes.append(max_heart_rate)
    attributes.append(exercise)
    attributes.append(oldpeak)
    attributes.append(slope)
    attributes.append(vessels)
    attributes.append(thal)

    #np.random.shuffle(data)
    k_fold=5
    fold_size=kfold(data,k_fold)
    test_result=[]
    for i in range(k_fold):
        test_data=data[int(i*fold_size):int((i+1)*fold_size),:]
        train_data=np.delete(data, range(int(i*fold_size),int((i+1)*fold_size)), 0)
        decision_tree=train(train_data,attributes)
        for toTest in test_data:
            test_result.append(decision_tree.predict(toTest))
    ##print fold_size
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(test_result)):
        if test_result[i]==data[i,13] and data[i,13]==1:
            TP+=1
        if test_result[i]==data[i,13] and data[i,13]==-1:
            TN+=1
        if test_result[i]==-1 and data[i,13]==1:
            FN+=1
        if test_result[i]==1 and data[i,13]==-1:
            FP+=1

    print str(TP) + "  "+ str(FN)
    print str(FP) + "  "+ str(TN)
main()
