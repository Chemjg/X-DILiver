# -*- coding: utf-8 -*-

# CLIENT. Python 3.

import requests
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta


KEY_DESCRIPTORS = "Descriptors"
KEY_ECFP = "ECFP"
KEY_PFP= "PFP"

HOST = 'http://165.194.60.198:9003'

STATUS_DONE = 'Done'
STATUS_ERROR = 'Error'

CHEMICAL_FORMAT = 'smiles'
# 사용가능한 포맷: smiles, mdl, cml

def makeURL(smiles):
    global HOST, CHEMICAL_FORMAT
    return HOST + '/query/' + CHEMICAL_FORMAT + '/' + ','.join(smiles)
    
def sendJob(query):
    
    
    res = requests.get(query)
    
    #print(res.text)
    
    data = res.json()
    
    if data['Status'] == STATUS_ERROR:
        # Error
        print ('Error : ' + data['Message'])
        return data['Message']
    elif data['Status'] == STATUS_DONE:
        # Done
        print ('Done successfully.')
        #ret = base64.b64decode( data['Result'] )
        #de = bz2.decompress(ret)
        return data['Result'] # dict
    else:
        print('Unknown status: ' + data['Status'])
        return data['Message']
        





def loadSmiles(fname):
    box={}
    f=open(fname,'r', encoding='utf-8')
    for s in f.readlines():
        s = s.strip()
        if len(s) == 0: continue
        box[s]=None
    f.close()
    return list(box)


def saveResult(fname, data):
    
    #print ("Saving result onto " + fname)
    
    if data is not None:
        
        #print(data)
        
        f=open(fname, 'w')
        f.write(data)
        f.close()
        
        #print(data)    
    
    print ('Saved into : ' + fname)
    
def run():
    
    smiles = loadSmiles('example.txt')
    query = makeURL(smiles)
    result = sendJob(query)
    
    if result != "None":
        
        #print (result)
    
        desc = result[KEY_DESCRIPTORS]
        ecfp = result[KEY_ECFP]
        pfp = result[KEY_PFP]
        
        saveResult('result_desc.txt', desc)
        saveResult('result_ecfp.txt', ecfp)
        saveResult('result_pfp.txt', pfp)

    

def diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a) 
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)
    
if __name__ == '__main__':
    
    t_a = datetime.now()
    
    run()    
    #cancel('t6qkcXp7YvMAcrvxmcJy')
    
    t_b = datetime.now()
    
    print('Elapsed time: ', diff(t_a,t_b))
    
    