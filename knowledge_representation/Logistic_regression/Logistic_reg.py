# This code will work with this dataset
# https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
# Assignment 2 for knowledge represetation course
import csv
from numpy import genfromtxt
import numpy as np
import math
import resource, sys
import random
import matplotlib.pyplot as plt
class LogisticRegression:
    def __init__(self,data,results,categorical_fields,categorical_options,stocastic_subset_size,regulized):
        self.regulized=regulized
        self.categorical_fields=categorical_fields
        self.categorical_options=categorical_options
        self.added_attributes=0;
        if self.regulized==0:
            self.train_percent=0.65
            self.validation_percent=0.
        elif self.regulized==1:
            self.train_percent=0.6
            self.validation_percent=0.2
        self.stocastic_subset_size=stocastic_subset_size
        self.data=data
        self.setup_categoricals()
        self.data=self.data*1./np.max(self.data,axis=0)

        self.test_data=self.data[len(self.data)*(self.train_percent+self.validation_percent):]
        self.validation_data=self.data[len(self.data)*self.train_percent:len(self.data)*(self.train_percent+self.validation_percent)]
        self.data=self.data[0:len(self.data)*self.train_percent]
        self.results=results[0:len(data)*self.train_percent]
        self.test_results=results[len(data)*(self.train_percent+self.validation_percent):]
        self.validation_results=results[len(data)*self.train_percent:len(data)*(self.train_percent+self.validation_percent)]

        self.number_of_parameters= len(data[0]);
        self.number_of_samples= len(data);
        self.learning_rate=0.015

        self.w0=0.0
        self.weights=np.zeros((self.number_of_parameters+self.added_attributes,), dtype=np.float)
        #print len(self.weights)
    def setup_categoricals(self):
        for i in self.categorical_fields:
            data_left=self.data[:,:(i-1+self.added_attributes)]

            data_right=self.data[:,(i+self.added_attributes):]
            data_to_convert=self.data[:,i-1+self.added_attributes]
            data_converted=[]
            for j in range(len(data_to_convert)):
                toAdd=np.zeros((len(self.categorical_options[self.categorical_fields.index(i)])), dtype=np.float)
                toAdd[self.categorical_options[self.categorical_fields.index(i)].index(data_to_convert[j])]=1
                data_converted.append(toAdd.tolist())

            self.added_attributes+=len(self.categorical_options[self.categorical_fields.index(i)])-1
            self.data=np.concatenate((data_left,data_converted, data_right), axis=1)
    def optimize_regulizer(self,isBatch):
        values=[0.1,0.5,1,2,10,1000,100000]
        mean_error=[]
        for i in values:
            toPlot_train_error=[]
            toPlot_validation_error=[]
            self.w0=0.0
            self.weights=np.zeros((self.number_of_parameters+self.added_attributes,), dtype=np.float)
            iteration=0
            while(iteration<500):
                if isBatch==1:
                    self.update_batch(i)
                else:
                    self.update_stochastic(i)
                toPlot_validation_error+=[self.validation_error()]
                toPlot_train_error+=[self.train_error()]
                iteration+=1

            plt.plot(range(iteration), toPlot_validation_error, 'r--', range(iteration), toPlot_train_error)
            plt.axis([0, 500, 0.0, 0.5])
            plt.ylabel('Error')
            plt.xlabel('iterations')
            if isBatch==1:
                plt.savefig('train_test_error_batch_'+str(i)+'.png')
            else:
                plt.savefig('train_test_error_stochastic_'+str(i)+'.png')
            #plt.axis([0, 1000, 0, 0.3])
            #plt.show()
            plt.close()
            mean_error+=[np.mean(toPlot_validation_error)]

        #print mean_error
        return values[mean_error.index(min(mean_error))]

    def update_stochastic(self,regulization_term):
        sum=0
        weights_temp=np.zeros((self.number_of_parameters+self.added_attributes,), dtype=np.float)
        w0_temp=0.0
        chosen_subset=random.sample(range(0, len(self.data)), self.stocastic_subset_size)
        for j in chosen_subset:
            sum+=self.results[j]-self.calc_logistic_function(self.data[j,:])

        w0_temp=self.learning_rate*sum

        #update_value=np.zeros(len(self.weights), dtype=np.float)
        #for j in range(len(self.data)):
        #    update_value+=(self.results[j]*self.data[j,:])/(1+np.exp(self.results[j]*(self.w0+self.calc_logistic_function(self.data[j,:]))))
        for i in range(self.number_of_parameters+self.added_attributes):
            sum=0
            for k in chosen_subset:
                sum+=self.data[k,i]*(self.results[k]-self.calc_logistic_function(self.data[k,:]))

            weights_temp[i]=sum - self.weights[i]/regulization_term
        #print self.weights
        self.weights+=self.learning_rate*weights_temp
        self.w0+=w0_temp*self.learning_rate
        #print "weights  "+str(weights_temp
    def update_batch(self,regulization_term):
        sum=0
        weights_temp=np.zeros((self.number_of_parameters+self.added_attributes,), dtype=np.float)
        w0_temp=0.0
        for j in range(len(self.data)):
            sum+=self.results[j]-self.calc_logistic_function(self.data[j,:])

        w0_temp=self.learning_rate*sum

        #update_value=np.zeros(len(self.weights), dtype=np.float)
        #for j in range(len(self.data)):
        #    update_value+=(self.results[j]*self.data[j,:])/(1+np.exp(self.results[j]*(self.w0+self.calc_logistic_function(self.data[j,:]))))
        for i in range(self.number_of_parameters+self.added_attributes):
            sum=0
            for k in range(len(self.data)):
                sum+=self.data[k,i]*(self.results[k]-self.calc_logistic_function(self.data[k,:]))

            weights_temp[i]=sum - self.weights[i]/regulization_term

        self.weights+=self.learning_rate*weights_temp
        self.w0+=w0_temp*self.learning_rate
        #print "weights  "+str(weights_temp)
    def calc_logistic_function(self, row):
        dot_product=np.dot(row,self.weights)
        #print "overflow::: "+str(dot_product+self.w0)
        return np.exp(dot_product+self.w0)/(1+np.exp(dot_product+self.w0))
    def dot_product(self, row):
        return np.dot(row,self.weights)
    def calc_log_ratio(self,sample):
        return self.dot_product(sample) + self.w0
    def calc_log_likelihood(self):
        sum=0
        #for i in range(len(self.data)):
        #    sum+=np.log(1+np.exp(-self.results[i]*(self.calc_logistic_function(self.data[i,:])+self.w0)))
        for i in range(len(self.data)):
            sum+=self.results[i]*(self.dot_product(self.data[i,:])+self.w0)
            sum-=np.log(1+np.exp(self.dot_product(self.data[i,:])+self.w0))
        return sum
    def test(self):
        TP=0
        TN=0
        FP=0
        FN=0
        total_error=0
        for i in range(len(self.test_data)):
            ratio=self.calc_logistic_function(self.test_data[i])
            total_error+=abs(self.test_results[i]-ratio)
            if ratio<0.5 and self.test_results[i]==0 :
                TN+=1
            elif ratio<0.5 and self.test_results[i]==1:
                FN+=1
            elif ratio>=0.5 and self.test_results[i]==1 :
                TP+=1
            elif ratio>=0.5 and self.test_results[i]==0 :
                FP+=1

        precision=TP/float(TP+FP)
        recall=TP/float(TP+FN)
        print str(TP) + "  "+ str(FN)
        print str(FP) + "  "+ str(TN)
        print "accuracy:"+ str((TP+TN)/float(TP+FP+FN+TN))
        print "precision:"+ str(precision)
        print "recall:"+ str(recall)
        print "F1-measure:"+ str(2*precision*recall/float(precision+recall))
        return total_error*1./len(self.test_data)
        #return (TP+TN)*1./(TP+TN+FP+FN)
    def train_error(self):
        TP=0
        TN=0
        FP=0
        FN=0
        total_error=0
        for i in range(len(self.data)):
            ratio=self.calc_logistic_function(self.data[i])
            total_error+=abs(self.results[i]-ratio)
            if ratio<0.5 and self.results[i]==0 :
                TN+=1
            elif ratio<0.5 and self.results[i]==1:
                FN+=1
            elif ratio>=0.5 and self.results[i]==1 :
                TP+=1
            elif ratio>=0.5 and self.results[i]==0 :
                FP+=1

        #print "Accuracy "+str((TP+TN)*1./(TP+TN+FP+FN))
        #print "Precision "+str(TP*1./(TP+FP))
        #print "Recall "+str(TP*1./(TP+FN))
        #return (FP+FN)*1./len(self.data)
        return total_error*1./len(self.data)
        #return (TP+TN)*1./(TP+TN+FP+FN)
    def validation_error(self):
        TP=0
        TN=0
        FP=0
        FN=0
        total_error=0
        for i in range(len(self.validation_data)):
            ratio=self.calc_logistic_function(self.validation_data[i])
            total_error+=abs(self.validation_results[i]-ratio)
            if ratio<0.5 and self.validation_results[i]==0 :
                TN+=1
            elif ratio<0.5 and self.validation_results[i]==1:
                FN+=1
            elif ratio>=0.5 and self.validation_results[i]==1 :
                TP+=1
            elif ratio>=0.5 and self.validation_results[i]==0 :
                FP+=1
        precision=TP/float(TP+FP)
        recall=TP/float(TP+FN)
        print str(TP) + "  "+ str(FN)
        print str(FP) + "  "+ str(TN)
        print "accuracy:"+ str((TP+TN)/float(TP+FP+FN+TN))
        print "precision:"+ str(precision)
        print "recall:"+ str(recall)
        print "F1-measure:"+ str(2*precision*recall/float(precision+recall))
        #print "Accuracy "+str((TP+TN)*1./(TP+TN+FP+FN))
        #print "Precision "+str(TP*1./(TP+FP))
        #print "Recall "+str(TP*1./(TP+FN))
        #return (TP+TN)*1./(TP+TN+FP+FN)
        #return (FP+FN)*1./len(self.validation_data)
        return total_error*1./len(self.validation_data)
def batch_gradient_normal():
    data = genfromtxt('heart.dataset', delimiter=' ')
    data[:,13]=data[:,13]-1
    np.random.shuffle(data)
    classifier=LogisticRegression(data[:,0:13],data[:,13],[3,7,13],[[1,2,3,4],[0,1,2],[3,6,7]],0,0)
    #print data[:,0:12]

    toPlot_likelihood=[]
    toPlot_train_error=[]
    toPlot_test_error=[]
    iteration=0
    previous_likelihood=-1000;
    current_likelihood=classifier.calc_log_likelihood();
    while(abs(current_likelihood-previous_likelihood)>0.05):
        classifier.update_batch(1000000)
        toPlot_likelihood+=[current_likelihood]

        previous_likelihood=current_likelihood
        current_likelihood=classifier.calc_log_likelihood();
        print ""+str(current_likelihood-previous_likelihood)+"   "+str(previous_likelihood)
        toPlot_test_error+=[classifier.test()]
        toPlot_train_error+=[classifier.train_error()]
        iteration+=1

    plt.plot(range(iteration), toPlot_likelihood,lw=2)
    plt.ylabel('log likelihood')
    plt.xlabel('iterations')
    plt.savefig('likelihood_plot_batch.png')
    plt.show()
    plt.plot(range(iteration), toPlot_test_error, 'r--', range(iteration), toPlot_train_error)
    plt.ylabel('Error')
    plt.xlabel('iterations')
    plt.savefig('train_test_error_batch.png')
    plt.show()
    classifier.test()
def stochastic_gradient_normal():
    data = genfromtxt('heart.dataset', delimiter=' ')
    data[:,13]=data[:,13]-1
    np.random.shuffle(data)

    for num in range(1,100,20):
        classifier=LogisticRegression(data[:,0:13],data[:,13],[3,7,13],[[1,2,3,4],[0,1,2],[3,6,7]],num,0)
        toPlot_likelihood=[]
        toPlot_train_error=[]
        toPlot_test_error=[]
        iteration=0
        previous_likelihood=-1000;
        current_likelihood=classifier.calc_log_likelihood();
        while(iteration<200):
            classifier.update_stochastic(10000000000000000000)
            toPlot_likelihood+=[current_likelihood]

            previous_likelihood=current_likelihood
            current_likelihood=classifier.calc_log_likelihood();
            print ""+str(current_likelihood-previous_likelihood)+"   "+str(previous_likelihood)
            toPlot_test_error+=[classifier.test()]
            toPlot_train_error+=[classifier.train_error()]
            iteration+=1

        plt.plot(range(iteration), toPlot_likelihood,lw=2)
        plt.ylabel('likelihood')
        plt.xlabel('iterations')
        plt.savefig('likelihood_plot_stochastic_'+str(num)+'.png')
        #plt.show()
        plt.close()
        plt.plot(range(iteration), toPlot_test_error, 'r--', range(iteration), toPlot_train_error)
        plt.ylabel('Error')
        plt.xlabel('iterations')
        plt.axis([0, 200, 0, 0.5])
        plt.savefig('train_test_error_stochastic_'+str(num)+'.png')
        #plt.show()
        plt.close()
        #classifier.test()
        #print toPlot_test_error

    #classifier.train_error()
def stochastic_gradient_regulized():
    data = genfromtxt('heart.dataset', delimiter=' ')
    data[:,13]=data[:,13]-1
    np.random.shuffle(data)
    classifier=LogisticRegression(data[:,0:13],data[:,13],[3,7,13],[[1,2,3,4],[0,1,2],[3,6,7]],40,1)
    selected_reg_value=classifier.optimize_regulizer(0);
    print selected_reg_value
def batch_gradient_regulized():
    data = genfromtxt('heart.dataset', delimiter=' ')
    data[:,13]=data[:,13]-1
    np.random.shuffle(data)
    classifier=LogisticRegression(data[:,0:13],data[:,13],[3,7,13],[[1,2,3,4],[0,1,2],[3,6,7]],0,1)
    selected_reg_value=classifier.optimize_regulizer(1);
    print selected_reg_value
    #print data[:,0:12]
    #toPlot_likelihood=[]
    #toPlot_train_error=[]
    #toPlot_test_error=[]
    #iteration=0
    #previous_likelihood=-1000;
    #current_likelihood=classifier.calc_log_likelihood();
    #while(abs(current_likelihood-previous_likelihood)>0.15):
    #    classifier.update_batch()
    #    toPlot_likelihood+=[current_likelihood]

    #    previous_likelihood=current_likelihood
    #    current_likelihood=classifier.calc_log_likelihood();
    #    print ""+str(current_likelihood-previous_likelihood)+"   "+str(previous_likelihood)
    #    toPlot_test_error+=[classifier.test()]
    #    toPlot_train_error+=[classifier.train_error()]
    #    iteration+=1

    #plt.plot(range(iteration), toPlot_likelihood,lw=2)
    #plt.show()
    #plt.plot(range(iteration), toPlot_test_error, 'r--', range(iteration), toPlot_train_error)
    #plt.show()
    #classifier.test()
def main():
    if sys.argv[1]=="batch":
        if sys.argv[2]=="normal":
            batch_gradient_normal()
        elif sys.argv[2]=="reg":
            batch_gradient_regulized()
    elif sys.argv[1]=="stochastic":
        if sys.argv[2]=="normal":
            stochastic_gradient_normal()
        elif sys.argv[2]=="reg":
            stochastic_gradient_regulized()
main()
print "Done!"
