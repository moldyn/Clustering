#!/usr/bin/python2.7
#
#
# mpp.py
#
# calculate most probable path.
#
#
#
# Copyright (c) 2013, 2014, Abhinav Jain, Florian Sittel, Matthias Ernst, Gerhard Stock.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

"""
@todo:
1) parallelize as much subroutines as possible
"""

import argparse
from numpy import *
from scipy.sparse import *
from scipy.io import *
import gzip
import glob
import warnings
import math as m
from os.path import *
import collections
warnings.simplefilter('ignore', SparseEfficiencyWarning)

def main():
    parser = argparse.ArgumentParser(description='Most probable path clustering script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tt', '-t', dest='tt', default='tt_init.txt', help='Trajectory file (containing index column and data column; can be gzipped)')
    parser.add_argument('--centers', '-c', dest='centers', default='centers_init.txt', help='File containing microstate centers')
    #parser.add_argument('--ttb', dest='ttb', default='ttb.txt', help='trajectory pieces')
    parser.add_argument('--ttb', dest='ttb', default=None, help='OPTIONAL: File containing assignment of trajectory pieces (same length as trajectory file). If not specified or not existing, treat whole trajectory as one.')
    parser.add_argument('--tstep', '-l', '--lagtime',  dest='tstep', type=int, default=10, help='Lagtime')
    parser.add_argument('--qstep', dest='qstep', type=float, default=0.01, help='OPTIONAL: Step for increasing qmin (change with care!)')
    parser.add_argument('--pop-thresh', "-p", dest='pop_thresh', type=int, default=0, help='OPTIONAL: Threshold for keeping highly populated states and not lumping them (kind of "constrained" clustering, usually do as second step after a clustering without thresholds)')
    parser.add_argument('--ram', "-m", dest='max_ram', type=float, default=3.5, help='OPTIONAL: Maximum amount of RAM (in GB) to use for holding data')
    parser.add_argument('--zip', "-z", dest='zip', action='store_true', default=False, help='OPTIONAL: GZip output text files')
    parser.add_argument('--saveall', "-s", dest='saveall', action='store_true', default=False, help='OPTIONAL: Save all data generated. If not set, future map, source-sink-map, mpp and transition probability matrix will not be saved.')

    args = parser.parse_args()
    
    print " -> MPP - Most probable path clustering <-"
    
    tstep = args.tstep
    print " - Using lagtime %i" % tstep
    if args.pop_thresh>1:
        print " - Will keep states with population of at least %i artificially stable" % args.pop_thresh
    
    i_start=0
    q_start=0.01
    if args.zip:
        file_suffix=".gz"
    else:
        file_suffix=""
    
    #check if there is an aborted run in current folder
    tt_resume, centers_resume, tt_current, centers_current=check_aborted_run()
    if tt_resume!="":
        q_cur, i_cur=round(float(tt_resume[3:7]), 2), int(tt_resume[8:10])
        q_start, i_start=round(float(tt_current[3:7]), 2), int(tt_current[8:10])
        
        #check if there is a population file "after" the last run. If so, check if it contains only one population (i.e. if algorithm did converge before)
        recover_files=["pop_%1.2f_%02d.txt" % (q_start, i_start), "pop_%1.2f_%02d.txt.gz" % (q_start, i_start), "pop_%1.2f_%02d.txt.bz2" % (q_start, i_start)]
        for file in recover_files:
            if exists( file ):
                pop=genfromtxt(file, unpack=True)
                if len(pop.shape)==1:
                    #there is only one population left, so nothing to do. Don't resume, but exit...
                    print "\n!! Found complete clustering run. Will not redo but exit."
                    return
        if q_cur==q_start and i_cur==i_start:
            i_start+=1
        
        print " - Recovering from trajectory file %s and center file %s..." % (tt_resume, centers_resume)
    
        tt = array(readFile( tt_resume ),dtype=int)
        centers = array(readFile( centers_resume ))
    else:
        # begin input
        print " - Reading trajectory file %s and center file %s..." % (args.tt, args.centers)
        tt = array(readFile( args.tt ),dtype=int)
        if len(tt.shape)==1 or tt.shape[1]==1:
            #obviously, we have data without index, so we create the index to be consistent
            print "  => Initial trajectory file had no indexing. Adding index..."
            tt=array([arange(len(tt)), tt], dtype=int).T
        centers = array(readFile( args.centers ))
        if not (centers[0,0]==0 or centers[0,0]==1) and not (centers[-1,0]==centers.shape[1]-1 or centers[-1,0]==centers.shape[1]):
            print "  => Initial center file had no indexing. Adding index..."
            centers=insert(centers.T, 0, arange(centers.shape[0]), axis=0).T
        
    ttb=get_ttb(args.ttb, tt)
    nclus = 1+max(tt[:,1])
    pop = free_energy(tt,nclus)
    #check if there are empty clusters and we therefore need to drop them and renumber everythin
    if len(where(pop.T[1]==0)[0])>0:
        print "  ! need to renumber states as there are empty states in initial clustering"
        dat=tt.T[1]
        
        centers=centers[where(pop.T[1]>0)]
        pop=pop[where(pop.T[1]>0)]
        for index, (state, curpop) in enumerate(pop):
            if index!=state:
                dat[where(dat==state)]=index
        tt[:,1]=dat
        nclus = 1+max(dat)
        pop = free_energy(tt,nclus)     #calculate populations again for renumbered trajectory (could be done above, but this way it's failsave)
        savetxt("tt_init.txt%s" % file_suffix, tt, fmt="%i")
        savetxt("centers_init.txt%s" % file_suffix, centers, fmt="%f")
    
    print " - Counting initial transitions..."
    tcount,tprob = transitions(tt,tstep,ttb,nclus, args.max_ram)
    
    if tt_resume!="":
        mmwrite("tcount_%0.2f_%02d.mtx" % (q_cur,i_cur), tcount)
        mmwrite("tprob_%0.2f_%02d.mtx" % (q_cur,i_cur), tprob)
        savetxt("pop_%1.2f_%02d.txt%s" % (q_cur,i_cur, file_suffix), pop, fmt="%i")
        print " - Resuming algorithm at qmin %f and iteration %i..." % (q_start, i_start)
    else:        
        mmwrite("tcount_init.mtx",tcount)
        mmwrite("tprob_init.mtx",tprob)
        savetxt("pop_init.txt%s" % file_suffix, pop, fmt="%i")
        print " - Starting algorithm..."
    
    #check if we can start at a higher q_start by first transition matrix
    q_step=args.qstep
    dia=[tprob[index,index] for index in xrange(tprob.shape[1])]
    if min(dia)>q_start:
        print "  + Lowest initial self-transition probability (%6.4f) is higher than starting q_min (%4.2f)." % (min(dia), q_start)
        q_start=(int(min(dia) / q_step))*q_step
        print "    Raising q_start to %4.2f" % q_start
        i_start=0       #reset iteration counter

    q_min=q_start
    while q_min<1.00+q_step and len(pop)>1:
        #convergence criteria: either, we found one state (i.e. pop=1) or we reached a qmin of (at least) 1
        if q_min>1.00:      #we won't go further than 1.00, even if we overshooted by adding q_step
            q_min=1.00
            
        print " -> q_min=%f" % q_min
        for i in range(i_start, 30):
            qmin_converged=False
            #test if there is a "gap" in q_min, meaning the lowest self-transition probability is higher than current q_min
            #if so, skip intermediate values and jump directly to next reasonable value (rounded up of lowest self-tr.prob to q_step). This saves time at no loss
            dia=array([tprob[index,index] for index in xrange(tprob.shape[1])])
            if min(dia)>q_min and q_min>q_start:
                print "  + Lowest self-transition probability (%6.4f) is higher than current q_min (%4.2f). Raising q_min" % (min(dia), q_min)
                q_min=(int(min(dia) / q_step))*q_step
                i_start=0       #reset iteration counter
                break           #and break the iteration loop, so go on with new value for q_min
            
            # immediate future
            fmap = future(tprob, centers, tt, pop, nclus, q_min, args.pop_thresh)
            if args.saveall:
                savetxt("fmap_%0.2f_%02d.txt%s" % (q_min,i, file_suffix), fmap, fmt="%i") 
            # most probable path
            mpp = mppath(fmap,nclus)
            if args.saveall:
                writeList(mpp,"mpp_%0.2f_%02d.txt%s" % (q_min,i, file_suffix))
            # source sink map
            ssmap = pathsink(pop, tprob, mpp, centers, nclus, q_min)
            if args.saveall:
                savetxt("ssmap_%0.2f_%02d.txt%s" % (q_min,i, file_suffix), ssmap, fmt="%i")
            # lumping states
            basins = basin(ssmap)
            writeList(basins,"basins_%0.2f_%02d.txt%s" % (q_min,i, file_suffix))
            # lumped centers
            centers = centfun(basins, centers, pop, False)
            savetxt("centers_%0.2f_%02d.txt%s" % (q_min,i, file_suffix), centers, fmt="%9.5f")
            # lumped trajectory
            tt = update_tt(tt,ssmap,nclus)
            savetxt("tt_%0.2f_%02d.txt%s" % (q_min,i, file_suffix), tt, fmt="%i")
            # lumped transition matrix
            nclus = 1+max(tt[:,1])
            tcount,tprob = transitions(tt,tstep,ttb,nclus, args.max_ram)
            mmwrite("tcount_%0.2f_%02d.mtx" % (q_min,i), tcount)
            if args.saveall:
                mmwrite("tprob_%0.2f_%02d.mtx" % (q_min,i), tprob)
            
            if len(basins) == len(pop):
                #iteration is converged, if every state goes to exactly one microstate (i.e. is stable)
                qmin_converged=True
            # population

            pop = free_energy(tt,nclus)
            savetxt("pop_%4.2f_%02d.txt%s" % (q_min,i, file_suffix), pop, fmt="%i")
            # convergence of algo: are we left with one microstate?
            if len(pop) == 1:
                print "converged to one state... finished!"
                break
            
            # convergence of iterations
            if qmin_converged:
                break
        # convergence of algo
        if len(pop)==1:     
            break
        if args.pop_thresh>1 and (pop.T[1]>=args.pop_thresh).all():      #all populations have reached the (non-zero) population threshold. No further lumping will happen, so exit
            print "All remaining states are larger than population threshold... stopping algorithm."
            break
        i_start=0
        
        #increase q_min to next step (round it to avoid floating precision errors)
        q_min=round(q_min+q_step,2)
        
    print " -> Done"
    return

def get_ttb(filename, trajectory):
    """
    either reads trajectory pieces from filename or otherwise creates array of 1 with same length as trajectory 
    """
    if filename and exists(filename):
        #use ttb file if it exists
        print "Using trajectory breaks stored in file %s" % filename
        ttb = array(readFile( filename ),dtype=int)
    else:
        #no ttb file specified or not existent: use just a series of 1
        ttb = ones_like(trajectory)
    return ttb

def readFile( filename, delim=" ", verbose=False):
    """
    filename : file to read from
    delim : delimiter
    """
    buf = []
    if verbose:
        print "Reading file %s" % filename
        
    name, ext=splitext(filename)
    if ext==".gz":
        fh = gzip.open(filename, 'r')
    else:
        fh = open(filename, 'r')
    
    try:
        for line in fh:
            #line = line.strip().split( delim )
            line=line.strip().split()   #just split it, regardless of delimiter
            buf.append( map(float, line) )
    finally:
        fh.close()
    return buf

def writeList(m, filename):
    """
    m : list variable to write
    filename : file to write into
    """
    name, ext=splitext(filename)
    if ext==".gz":
        fh = gzip.open(filename, 'w')
    else:
        fh = open(filename, 'w')
    try:
      for row in m:
        row = " ".join( map(str, row) )
        fh.write( row + "\n" )
    finally:
      fh.close()
    return

def free_energy(tt,nclus):
    """
    calculates population of microstates
    tt : trajectory list
    """
    pop = zeros(shape=(nclus,2),dtype=int)
    pop[:,0]=range(nclus)
    for i in range(len(tt[:,1])):
        pop[tt[i,1],1] += 1
    if sum(pop[:,1]) != len(tt[:,1]):
        raise Exception("Error, trajectory length %i does not match sum of all populated states (%i). Something went wrong, this should not happen!" % (len(tt[:,1]), sum(pop)))
    return pop.astype(int)

def dis(src,trg,centers):
    """
    calculate distance between centers of source and target microstate
    src : source, and
    trg : target
    """
    d = (sum((centers[trg,:]-centers[src,:])**2))**0.5
    return round(d,6)

def future(tprob, centers, tt, pop, nclus, q_min, pop_thresh):
    """
    calculate immediate future of microstates at single timestep
    tprob : probability matrix (see transitions1)
    tt : trajectory
    pop : population
    nclus : number of microstates
    q_min : metastability cut-off
    pop_thresh: threshold for keeping highly populated states artificially stable
    """
    fmap = array([[0]*2]*nclus)
    fmap[:,0]=range(nclus)
    for i in range(nclus):
        row = tprob.getrow(i)
        
        if tprob[i,i] >= q_min:
            futu = [i]
            """
            #This is "old style": do not care if metastability of state is higher than qmin but lump it to its most probable path anyway
            row_data = row.data
            row_id = row.indices
            futu = row_id[where(row_data == max(row_data))]      
            """
        elif tprob[i,i] < q_min:
            row[0,i] = 0
            row_data = row.data
            row_id = row.indices
            futu = row_id[where(row_data == max(row_data))]
        if pop_thresh>1 and pop[i][1]>=pop_thresh:
            #first, check if threshold for population is set and if so, if this state has such high population it shall not be lumped            
            #print i, pop[i][1], pop_thresh
            #print "Population of state %i is %i and thus above threshold of %i. Keeping this state.: "% (i, pop[i][1], pop_thresh)
            #if len(futu)>1 or futu!=[i]:
            if futu!=[i]:
                print "Keeping state %i artificially stable because of population %i. Future would have been: %s" % (i, pop[i][1], str(futu))
                futu = [i]
            
        if len(futu) == 1:
            sink = futu[0]
        if len(futu) > 1:
            p_futu = pop[futu,:]
            n_futu = p_futu[where(p_futu[:,1] == max(p_futu[:,1])),0][0]
            if len(n_futu) == 1:
                sink = n_futu[0]
            if len(n_futu) > 1:
                ssd = []
                for j in n_futu:
                    ssd.append(dis(i,j,centers))
                sink = n_futu[argmin(ssd)]
        fmap[i,1] = sink
    return fmap.astype(int)

def transitions(tt,tstep,ttb,nclus,max_ram_gb=3.5):
    """
    calculate transition matrix 
    tt         : trajectory 
    tstep      : lag time
    ttb        : breaks points if multiple
                 trajectories are concatenated
    nclus      : number of microstates
    max_ram_gb : maximum amount of RAM to use in GB. If too many clusters are given, a (slow) sparse matrix will be used instead of fast 2D arrays
    """
        
    #determine maximum number of items we can store in a quadratic numpy array
    max_n=int(sqrt(max_ram_gb*1024*1024*1024/dtype(int).itemsize)*0.98) #0.98 for overhead reasons 
    
    if nclus>max_n:
        tcount = lil_matrix((nclus,nclus), dtype=int)
    else:
        tcount = zeros((nclus,nclus), dtype=int)    #it's much faster accessing a numpy array than a sparse matrix

    transits = zip(tt[:-tstep,1],tt[tstep:,1]) # all possible transitions on lagtime of tstep
    valid_transits = [transits[index] for index in where(ttb[:-tstep] - ttb[tstep:] == 0)[0]] # removing incorrect transitions if there are multiple trajectories
    transits_list = collections.Counter(valid_transits).most_common()  # counting frequency of transitions
    for i in transits_list:    # updating transition frequency in transition count matrix
        tcount[i[0][0], i[0][1]] = i[1]
    

#    for i in range(len(tt[:,1])-tstep):
#        if ttb[i,0] == ttb[i+tstep,0]:
#            tcount[tt[i,1],tt[i+tstep,1]] += 1
#    
    if nclus<=max_n:    #convert it back to sparse matrix in the end
        tcount=lil_matrix(tcount)
    
    norm = spdiags(1./tcount.sum(1).T,0,*tcount.shape)
    tprob = norm * tcount
    return (tcount, tprob)

def mppath(fmap,nclus):
    """
    calculate most probable path of microstates
    fmap : immediate future of microstates
    nclus : number of microstates
    """
    mpp = []
    for i in range(nclus):
        j=i
        mpp.append( [] )
        while ((j in mpp[i])==False):
            mpp[i].append(j)
            j = fmap[j][1]
    return mpp



def pathsink(pop, tprob, mpp, centers,nclus, q_min):
    """
    calculate sink for most probable path of microstates
    pop : population
    tprob : transition probabilities
    mpp : most probable path of microstates
    centers : geometric centers of microstates
    """
    ssmap = array([[0]*2]*nclus)
    ssmap[:,0]=range(nclus)
    dprob = tprob.diagonal()
    for i in range(nclus):
        # preference for states with metastability > Q_min for calculation of psink, correcting for scenario when a state with metastability < Q_min has the highest free energy
        # however if none of the states satisfy metastability > Q_min then all are considered and ranked by their free-energy
        stable_sinks = where(dprob[mpp[i]] >= q_min)[0]
        if len(stable_sinks) > 1:
            mppi = array(mpp[i])[stable_sinks]
        if len(stable_sinks) == 1:
            mppi = array(mpp[i])[stable_sinks]
        else:
            mppi = mpp[i]
        psink = array(mppi)[where(pop[mppi,1] == max(pop[mppi,1]))]
        if len(psink) > 1:
            dsink = psink[where(dprob[psink] == max(dprob[psink]))]
            if len(dsink) == 1:
                psink = dsink
            if len(dsink) > 1:
                ssd = []
                for j in dsink:
                    ssd.append(dis(i,j,centers))
                psink = dsink[argmin(ssd)]
                
                #check if there are several states with equal population and same MPP
                for iters in dsink[1:]:
                    if set(mpp[iters]) != set(mppi):
                        break
                else:
                    #all states in the MPP have the same MPP, same population and same metastability, so set the same destination for all equally interconverting states (i.e. merge all)
                    ssmap[mppi,1] = i
                    psink = i
        ssmap[i,1] = psink
    return ssmap.astype(int)

def basin(ssmap):
    """
    Calculate basins, i.e., microstates converging into same sink
    ssmap : source sink mapping 
    """
    bas = unique(ssmap[:,1])
    basins = []
    for i in range(len(bas)):
        bas_merge =  append( append(array(i), array(bas[i])), where(ssmap[:,1] == bas[i])[0])
        basins.append(bas_merge.tolist())
    return basins

def centfun(basins, centers, pop, circ):
    """
    calculate geometric centers for merged microstates
    basins = microstates basins
    centers = geometric centers of microstates
    pop = microstate populations
    circ = if True: space is treated as circular(angles in degrees), 
           if False: space is treated as metric
    """
    if circ == False:
        newcents = zeros(shape=(len(basins),centers.shape[1]))
        for i in range(len(basins)):
            bmerge = basins[i][2:]
            fmerge = pop[bmerge,1]
            cmerge = centers[bmerge][:,1:]
            if len(bmerge) == 1:
                newcents[i] = append(i,cmerge)
            if len(bmerge) > 1:
                newcents[i] = append(i,array(sum(diag(fmerge) * matrix(cmerge),axis=0)/sum(fmerge)))
    if circ == True:
        newcents = zeros(shape=(len(basins),centers.shape[1]))
        for i in range(len(basins)):
            bmerge = basins[i][2:]
            fmerge = pop[bmerge,1]
            cmerge = centers[bmerge][:,1:]
            if len(bmerge) == 1:
                newcents[i] = append(i,cmerge)
            if len(bmerge) > 1:
                c = average(cos(radians(cmerge)),axis=0,weights=fmerge)
                s = average(sin(radians(cmerge)),axis=0,weights=fmerge)
                t = zeros(shape=c.shape)
                for j in range(len(c)):
                    t[j]=degrees(m.atan2(s[j],c[j]))
                newcents[i] = append(i,t)
    return newcents

def update_tt(tt,ssmap,nclus):
    """
    calculate updated trajectory of merged microstates
    tt : trajectory
    ssmap : source sink mapping
    nclus : microstates in trajectory
    """
    bas = unique(ssmap[:,1])
    ttmap = zeros(shape=(2,nclus))
    ttmap[0,:] = range(nclus)
    ttmap[1,bas] = range(len(bas))
    ssmap_new = array([[0]*nclus]*2)
    ssmap_new[0,:]=range(nclus)
    ssmap_new[1,:]=ttmap[1,ssmap[:,1]]
    ssmap_new = ssmap_new.transpose()
    ttnew = zeros(shape=tt.shape)
    ttnew[:,0]= range(tt.shape[0])
    ttnew[:,1]=ssmap_new[tt[:,1],1]
    return ttnew.astype(int)

def check_aborted_run():
    """
    checks if there is a previous and maybe aborted run in current directory
    """
    trajs=glob.glob("tt_[0-1].[0-9][0-9]_[0-9][0-9].*")     #if the naming scheme if the files is changed at saving time, change it here too!
    centers=glob.glob("centers_[0-1].[0-9][0-9]_[0-9][0-9].*")
    pops=glob.glob("pop_[0-1].[0-9][0-9]_[0-9][0-9].*")
    trajs.sort()
    centers.sort()
    pops.sort()
    
    if len(pops)==len(trajs) and len(trajs)==len(pops):
        """
        if len(pops)>0:
            print pops[-1]
            pop_last=numpy.loadtxt(pops[-1], dtype=int)
            if len(pop_last.shape)==1:
                print "Population converged"
            return "","","",""
            """
    
        if len(centers)>0 and len(trajs)>0:
            index=len(centers)-1 
            t, c=trajs[index], centers[index]
            
            offset=1
            #if the last population file has exactly one line, the calculation has converged and we do not want to resume from before
            pop_last=loadtxt(pops[-1], dtype=int)
            if len(pop_last.shape)==1:
                offset=0
            
            if t[3:7]==c[8:12] and t[8:10]==c[13:15]:
                return trajs[index-offset], centers[index-offset], t, c
    
    return "","","",""

if __name__ == "__main__":
    main()

