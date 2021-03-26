#!/bin/bash

#the output result is as follows: vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,
                                  #percentage,sOriginalDistance,sMinEdgetoProcess,sApprAttrValues,
                                  #sReduceExecution,sPartialGraphProcess,Error,executionTime

file1='commands_bellman.txt'
file2='graph_names.txt'
file3='../bellman_originaldistance.txt'

make -f Makefile_bellman || exit

while read line1;
do
while read line2;
do
echo $line2


if [ "$line2" == "1 0 0 0 0" ];
then
./run $line2 $line1 0 0 1
fi


iter=$(head -n 1 $file3)
maxEdge=$(sed -n '2p' $file3)


p=()

for t in {1..12};
do
myvar=$(((maxEdge / 1000) * t * t ))
p+=($myvar)
done


if [ "$line2" == "0 1 0 0 0" ];
then
for t in {0..11};
do
./run $line2 $line1 ${p[t]} $iter 1
done
fi


if [ "$line2" == "0 0 1 0 0" ];
then
./run $line2 $line1 0 $iter 1
fi


if [ "$line2" == "0 0 0 1 0" ];
then
for t in $(seq 0 $iter);
do
./run $line2 $line1 0 $t 1
done
fi


if [ "$line2" == "0 0 0 0 1" ];
then
for t in $(seq 0.1 .1 0.9)
do
./run $line2 $line1 0 $iter $t
done
fi

done < $file1
done < $file2


file1='commands_dijkstra.txt'
file2='graph_names.txt'
file3='../dijkstra_originaldistance.txt'

make -f Makefile_dijkstra || exit

while read line1;
do
while read line2;
do
echo $line2


if [ "$line2" == "1 0 0 0 0" ];
then
./run $line2 $line1 0 0 1
fi


iter=$(head -n 1 $file3)
maxEdge=$(sed -n '2p' $file3)


p=()

for t in {1..12};
do
myvar=$(((maxEdge / 1000) * t * t))
p+=($myvar)
done


if [ "$line2" == "0 1 0 0 0" ];
then
for t in {0..11};
do
./run $line2 $line1 ${p[t]} $iter 1
done
fi


if [ "$line2" == "0 0 1 0 0" ];
then
./run $line2 $line1 0 $iter 1
fi


if [ "$line2" == "0 0 0 1 0" ];
then
for t in $(seq 0 $iter);
do
./run $line2 $line1 0 $t 1
done
fi


done < $file1
done < $file2

