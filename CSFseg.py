import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import copy
import heapq

def connectToBoundary(label, classIdx, tolerance):
    neighbors=[]
    for i in range(-1, 2):
        for j in range(-1, 2):
            k=0
            neighbors.append((i,j,k))
                
    seen=set()
    
    position=[]
    heapq.heapify(position)

    island=0
    newLabel=np.zeros(label.shape)
    i, j, k=label.shape
    for z in range(k):
        for x in range(i):
            for y in range(j):
                
                if (label[x,y,z]==classIdx) and (x,y,z) not in seen:
                    island+=1
                    area=0
                    curIsland=set()
                    seen2=set()
                    seen.add((x,y,z))
                    curIsland.add((x,y,z))
                    heapq.heappush(position, (x,y,z))

                    connected=False
                    while position:
                        cur=heapq.heappop(position)
                    
                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
                            if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    

                            if (label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]==classIdx) and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                curIsland.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2], 0))

                    position2=[]
                    heapq.heapify(position2)
                    
                    
                    for cur in curIsland:
                        heapq.heappush(position2,(cur[0],cur[1],cur[2],0))
                        seen2.add(cur)
                    while position2:
                        cur=heapq.heappop(position2)
                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
                            if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    
                            if (label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]!=0) and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]) not in seen2 and cur[3]<tolerance:
                                seen2.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                heapq.heappush(position2, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2], cur[3]+1))
                            
                            elif label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]==0:
                                connected=True
                                break   
                    
                    if connected:

                        for (posX, posY, posZ) in curIsland: 

                            label[posX, posY, posZ]=3
                
def maxArea(label, classIdx, connectivity=8, findMax=True):
    neighbors=[]
    if connectivity==8:
        for i in range(-1, 2):
            for j in range(-1, 2):
                k=0
                neighbors.append((i,j,k))
    elif connectivity==4:
        neighbors=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]
        
    else:    
        return
                
    seen=set()
    
    position=[]
    heapq.heapify(position)
    islandDict={}
    
    maxArea=0
    maxPos=(0,0,0)
    island=0
    newLabel=copy.deepcopy(label)
    i, j, k=label.shape
    
    for z in range(k):
        for x in range(i):
            for y in range(j):
                
                if (label[x,y,z]==classIdx) and (x,y,z) not in seen:
                    island+=1
                    area=0
                    curIsland=set()
                    seen.add((x,y,z))
                    curIsland.add((x,y,z))
                    heapq.heappush(position, (x,y,z))

                    while position:
                        cur=heapq.heappop(position)
                        area+=1

                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
                            if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    

                            if label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]==label[x,y,z] and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                curIsland.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                    
                    islandDict[(x,y,z)]=frozenset(curIsland)
                    #  print(island, area)
                    
                    if findMax:
                        if area>maxArea: 
                            maxArea=area
                            maxPos=(x, y, z)

    # print("islandDict", islandDict)
    return islandDict[maxPos], maxArea, maxPos

def Connectivity(label, classIdx, targetIdx, refClass=1,connectivity=8):
    neighbors=[]
    if connectivity==8:
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbors.append((i,j))
    elif connectivity==4:
        neighbors=[(1,0),(-1,0),(0,1),(0,-1)]
        
    else:   
        return
                
    seen=set()
    
    island=0
    position=[]
    heapq.heapify(position)

    i, j=label.shape
    
    for x in range(i):
        for y in range(j):

            if (label[x,y]==refClass) and (x,y) not in seen:
                island+=1
                seen.add((x,y))
                heapq.heappush(position, (x,y))

                while position:
                    cur=heapq.heappop(position)
    
                    for neighbor in neighbors:

                        if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                        if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue

                        if label[cur[0]-neighbor[0],cur[1]-neighbor[1]]==classIdx and (cur[0]-neighbor[0],cur[1]-neighbor[1]) not in seen:
                            seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1]))
                            label[cur[0]-neighbor[0],cur[1]-neighbor[1]]=targetIdx
                            heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1]))

                

def numIsland(label,connectivity=8):
    neighbors=[]
    if connectivity==8:
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbors.append((i,j))
    elif connectivity==4:
        neighbors=[(1,0),(-1,0),(0,1),(0,-1)]
        
    else:
        
        return
               
    seen=set()
    
    island=0
    position=[]
    heapq.heapify(position)

    i, j=label.shape
    
    
    for y in range(j):
        for x in range(i-1,-1,-1):

            if (label[x,y]!=0) and (x,y) not in seen:
                
                if island==1:
                    if area>100: 
                        island+=1
                        break
                        
                    else: island=0

                if island==0:
                    island+=1
                    area=0
                    seen.add((x,y))
                    heapq.heappush(position, (x,y))
                    curIsland=set()
                    while position:
                        cur=heapq.heappop(position)
                        area+=1
                        curIsland.add(cur)
                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue

                            if label[cur[0]-neighbor[0],cur[1]-neighbor[1]]!=0 and (cur[0]-neighbor[0],cur[1]-neighbor[1]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1]))
                
                    maxArea=area
                    maxPos=curIsland
                        
    return island, maxArea, maxPos

def changeClassResult(segmentation):
    for x in range(segmentation.shape[0]):
        for y in range(segmentation.shape[1]):
            for z in range(segmentation.shape[2]):
                # Sub arch into 4.
                if segmentation[x,y,z]==3:
                    segmentation[x,y,z]=4
                #ventricles into class10
                elif segmentation[x,y,z]==1:
                    segmentation[x,y,z]=10

def saveImage(array, name, affine, header):
    
    #img = nib.Nifti1Image(array, np.eye(4))
    img = nib.Nifti1Image(array, affine, header)
    nib.save(img, name)  

def cutoff(label,max):

    neighbors=[(1,1,0),(0,1,0),(-1,1,0),(-1,0,0),(-1,-1,0),(0,-1,0),(1,-1,0),(1,0,0)]
    surpos = [3,3,3,3,3,3,3,3]
    i, j, k=label.shape

    for z in range(k):
        for x in range(i):
            for y in range(j):
                if label[x,y,z] ==2: 
                    nei = []
                    for neighbor in neighbors:
                        if x-neighbor[0]<0 or x-neighbor[0]>=i: continue
                        if y-neighbor[1]<0 or y-neighbor[1]>=j: continue
                        nei.append(label[x-neighbor[0], y-neighbor[1],z-neighbor[2]])
                    if nei == surpos :
                        label[x,y,z] = 3


def segVent(imgName, outputPath, resultName):
    print("resultname", resultName)
    im = nib.load(os.path.join(outputPath, resultName))
    result=im.get_fdata()
    affine = im.affine

    # Get the header
    header = im.header
    pixdim= header['pixdim']
    voxel_volume= pixdim[1]*pixdim[2]*pixdim[3]

    # Create directories if they do not exist
    #saveImage(result, os.path.join(outputPath, 'beforechangeClassResult'+resultName), affine, header)

    x,y,z= result.shape

    #voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
    print(f"Voxel volume: {voxel_volume} cubic mm")
    print("result shape", result.shape)
    print("result unique elements", np.unique(result))
    changeClassResult(result)

    #saveImage(result, os.path.join(outputPath, 'changeClassResult'+resultName), affine, header)

    #step 1: get subarachnoid connected to skull
    connectToBoundary(result, 4, tolerance=35)

    #saveImage(result, os.path.join(outputPath, 'connecttoBoundary'+resultName), affine, header)

    #step 3: get max area of remaining CSF
    island, Area, maxPos= maxArea(result, 10)
    for pos in island:
        result[pos]=1
    cutoff(result, maxPos)

    #saveImage(result, os.path.join(outputPath, 'cutoff'+resultName), affine, header)

    print("Maxpos, Area", maxPos, Area)
    for k in range(maxPos[2]-1,-1,-1):
        for i in range(x):
            for j in range(y):
                if result[i,j,k]==10 and result[i,j,k+1]==1:
                    result[i,j,k]=1

        Connectivity(result[:,:,k], 10, 1, refClass=1)

    for k in range(maxPos[2]+1, z):
        for i in range(x):
            for j in range(y):
                if result[i,j,k] ==10 and result[i,j,k-1]==1 :
                    result[i,j,k]=1
        Connectivity(result[:,:,k], 10, 1, refClass = 1)

    
    #saveImage(result, os.path.join(outputPath, 'connectivity'+resultName), affine, header)

    for k in range(z):
        for i in range(x):
            for j in range(y):
                if result[i,j,k]==10:
                    result[i,j,k]=3
    
    #check max pos of ventricle
    ventmaxArea = 0
    ventmaxPos = 0
    total_vent_voxels =0 
    for k in range(maxPos[2]-3,maxPos[2]+4):
        ventvoxel = 0
        for i in range(x):
            for j in range(y):
                if result[i,j,k]==1:
                    ventvoxel +=1
        total_vent_voxels += ventvoxel
        if ventvoxel > ventmaxArea :
            ventmaxArea = ventvoxel
            ventmaxPos = (i,j,k)

    print('middle of 7 slices :', maxPos[2])
       
    os.makedirs(os.path.join(outputPath + '/vent/'), exist_ok=True)
    saveImage(result, os.path.join(outputPath, 'vent/'+resultName), affine, header)
    print("result unique elements", np.unique(result))
    
    # Compute the voxel volume
    print(f"Voxel volume: {voxel_volume} cubic mm")
    total_vent_vol = float(total_vent_voxels*voxel_volume)

    return total_vent_vol, total_vent_voxels, ventmaxArea, ventmaxPos, result






















