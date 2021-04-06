#!/bin/bash

echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sApprAttrValues,sReduceExecution,sPartialGraphProcess,Error,executionTime" >> '../bellman_performance_results.csv'

file1='commands_bellman.txt'
file2='graph_names.txt'
file3='../bellman_originaldistance.txt'

perf_results='../performance_results.csv'
time_results='time_results.txt'


make -f Makefile || exit

while read line1;
do
echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>FILE_TO_PROCESS : ' $line1
while read line2;
do
echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SIGNALS : ' $line2


if [ "$line2" == "1 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 0 1 0 1
done
./run $line2 $line1 0 0 1 1 1
rm 'time_results.txt'
fi

iter=$(head -n 1 $file3)
maxEdge=$(sed -n '2p' $file3)


p=()

for t in {1..10};
do
myvar=$(((maxEdge / 1000) * t * t * t))
p+=($myvar)
done


if [ "$line2" == "0 1 0 0 0" ];
then
for t in {0..9};
do
for n in {0..9};
do
./run $line2 $line1 ${p[t]} $iter 1 0 1
done
./run $line2 $line1 ${p[t]} $iter 1 1 1
rm 'time_results.txt'
done
fi

if [ "$line2" == "0 0 1 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 1
done
./run $line2 $line1 0 $iter 1 1 1
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 1 0" ];
then
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 0 $t 1 0 1
done
./run $line2 $line1 0 $t 1 1 1
rm 'time_results.txt'
done
fi


p=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


if [ "$line2" == "0 0 0 0 1" ];
then
for t in {0..9}
do
for ((n=1; n<=10; n++))
do
./run $line2 $line1 0 $iter ${p[t]} 0 1
done
./run $line2 $line1 0 $iter ${p[t]} 1 1
rm 'time_results.txt'
done
fi

done < $file1
done < $file2


echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sApprAttrValues,sReduceExecution,sPartialGraphProcess,Error,executionTime" >> '../dijkstra_performance_results.csv'


file1='commands_dijkstra.txt'
file2='graph_names.txt'
file3='../dijkstra_originaldistance.txt'

make -f Makefile || exit


while read line1;
do
echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>FILE_TO_PROCESS : ' $line1
while read line2;
do
echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SIGNALS : ' $line2


if [ "$line2" == "1 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 0 1 0 2
done
./run $line2 $line1 0 0 1 1 2
rm 'time_results.txt'
fi


iter=$(head -n 1 $file3)
maxEdge=$(sed -n '2p' $file3)


p=()

for i in {1..10};
do
myvar=$(((maxEdge / 1000) * t * t *t))
p+=($myvar)
done


if [ "$line2" == "0 1 0 0 0" ];
then
for t in {0..9};
do
for n in {0..9};
do
./run $line2 $line1 ${p[t]} $iter 1 0 2
done
./run $line2 $line1 ${p[t]} $iter 1 1 2
rm 'time_results.txt'
done
fi


if [ "$line2" == "0 0 1 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 2
done
./run $line2 $line1 0 $iter 1 1 2
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 1 0" ];
then
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 0 $t 1 0 2
done
./run $line2 $line1 0 $t 1 1 2
rm 'time_results.txt'
done
fi


done < $file1
done < $file2


echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sApprAttrValues,sReduceExecution,sPartialGraphProcess,Error,executionTime" >> '../hybrid_performance_results.csv'


file1='commands_dijkstra.txt'
file2='graph_names.txt'
file3='../hybrid_originaldistance.txt'

make -f Makefile || exit


while read line1;
do
echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>FILE_TO_PROCESS : ' $line1
while read line2;
do
echo '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SIGNALS : ' $line2


if [ "$line2" == "1 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 0 1 0 3
done
./run $line2 $line1 0 0 1 1 3
rm 'time_results.txt'
fi


iter=$(head -n 1 $file3)
maxEdge=$(sed -n '2p' $file3)


p=()

for i in {1..10};
do
myvar=$(((maxEdge / 1000) * t * t *t))
p+=($myvar)
done


if [ "$line2" == "0 1 0 0 0" ];
then
for t in {0..8};
do
for n in {0..9};
do
./run $line2 $line1 ${p[t]} $iter 1 0 3
done
./run $line2 $line1 ${p[t]} $iter 1 1 3
rm 'time_results.txt'
done
fi


if [ "$line2" == "0 0 1 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 3
done
./run $line2 $line1 0 $iter 1 1 3
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 1 0" ];
then
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 0 $t 1 0 3
done
./run $line2 $line1 0 $t 1 1 3
rm 'time_results.txt'
done
fi


done < $file1
done < $file2