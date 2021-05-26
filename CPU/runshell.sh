#!/bin/bash

echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sApprAttrValues,sReduceExecution,sPartialGraphProcess,sAtomicMinBlock,sAtomicMaxBlock,sAtomicAddBlock,Error,executionTime" >> '../bellman_performance_results.csv'

file1='commands_bellman.txt'
file2='graph_names.txt'
file3='../bellman_originaldistance.txt'

perf_results='../performance_results.csv'
time_results='time_results.txt'


make -f Makefile || exit

while read line1;
do
echo '>>>>>>>>>>>>>>>'
echo '>>>>>>>>>>>>>>>'
echo 'FILE_TO_PROCESS : ' $line1
echo '>>>>>>>>>>>>>>>'
echo '>>>>>>>>>>>>>>>'
while read line2;
do
echo 'SIGNALS : ' $line2
echo '>>>>>>>>>>>>>>>'
echo '>>>>>>>>>>>>>>>'


if [ "$line2" == "1 0 0 0 0 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 0 1 0 1
done
./run $line2 $line1 0 0 1 1 1
rm 'time_results.txt'
fi

iter=$(head -n 1 $file3)


if [ "$line2" == "0 1 0 0 0 0 0 0 0" ];
then
for t in {1..16};
do
for n in {0..9};
do
./run $line2 $line1 $t $iter 1 0 1
done
./run $line2 $line1 $t $iter 1 1 1
rm 'time_results.txt'
done
fi


if [ "$line2" == "0 0 1 0 0 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 1
done
./run $line2 $line1 0 $iter 1 1 1
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 1 0 0 0 0 0" ];
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


if [ "$line2" == "0 0 0 0 1 0 0 0 0" ];
then
for n in {0..9}
do
./run $line2 $line1 0 $iter 0.6 0 1
done
./run $line2 $line1 0 $iter 0.6 1 1
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 0 0 1 0 0 0" ];
then
for n in {0..9}
do
./run $line2 $line1 0 $iter 0 0 1
done
./run $line2 $line1 0 $iter 0 1 1
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 0 0 0 1 0 0" ];
then
for n in {0..9}
do
./run $line2 $line1 0 $iter 0 0 1
done
./run $line2 $line1 0 $iter 0 1 1
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 0 0 0 0 1 0" ];
then
for n in {0..9}
do
./run $line2 $line1 0 $iter 0 0 1
done
./run $line2 $line1 0 $iter 0 1 1
rm 'time_results.txt'
fi


if [ "$line2" == "0 1 0 1 0 0 0 0 0" ];
then
for i in {1..16};
do
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 $i $t 1 0 1
done
./run $line2 $line1 $i $t 1 1 1
rm 'time_results.txt'
done
done
fi


if [ "$line2" == "0 1 0 0 1 0 0 0 0" ];
then
for i in {1..16};
do
for n in {0..9};
do
./run $line2 $line1 $i $iter 0.6 0 1
done
./run $line2 $line1 $i $iter 0.6 1 1
rm 'time_results.txt'
done
fi


if [ "$line2" == "0 0 0 1 1 0 0 0 0" ];
then
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 0 $t 0.6 0 1
done
./run $line2 $line1 0 $t 0.6 1 1
rm 'time_results.txt'
done
fi


if [ "$line2" == "0 1 0 1 1 0 0 0 0" ];
then
for i in {1..16};
do
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 $i $t 0.6 0 1
done
./run $line2 $line1 $i $t 0.6 1 1
rm 'time_results.txt'
done
done
fi

done < $file1
done < $file2



echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sApprAttrValues,sReduceExecution,sPartialGraphProcess,sAtomicMinBlock,sAtomicMaxBlock,sAtomicAddBlock,Error,executionTime" >> '../dijkstra_performance_results.csv'


file1='commands_dijkstra.txt'
file2='graph_names.txt'
file3='../dijkstra_originaldistance.txt'

make -f Makefile || exit


while read line1;
do
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'
echo 'FILE_TO_PROCESS : ' $line1
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'
while read line2;
do
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'
echo 'SIGNALS : ' $line2
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'


if [ "$line2" == "1 0 0 0 0 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 0 1 0 2
done
./run $line2 $line1 0 0 1 1 2
rm 'time_results.txt'
fi


iter=$(head -n 1 $file3)


if [ "$line2" == "0 1 0 0 0 0 0 0 0" ];
then
for t in {1..16};
do
for n in {0..9};
do
./run $line2 $line1 $t $iter 1 0 2
done
./run $line2 $line1 $t $iter 1 1 2
rm 'time_results.txt'
done
fi


if [ "$line2" == "0 0 1 0 0 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 2
done
./run $line2 $line1 0 $iter 1 1 2
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 1 0 0 0 0 0" ];
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


if [ "$line2" == "0 0 0 0 0 0 0 0 1" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 2
done
./run $line2 $line1 0 $iter 1 1 2
rm 'time_results.txt'
fi


if [ "$line2" == "0 1 0 1 0 0 0 0 0" ];
then
for i in {1..16};
do
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 $i $t 1 0 2
done
./run $line2 $line1 $i $t 1 1 2
rm 'time_results.txt'
done
done
fi

done < $file1
done < $file2



echo "vertexNum,edgeNum,iterationNum,maxEdgeDegree,minProcessEdge,percentage,sOriginalDistance,sMinEdgetoProcess,sApprAttrValues,sReduceExecution,sPartialGraphProcess,sAtomicMinBlock,sAtomicMaxBlock,sAtomicAddBlock,Error,executionTime" >> '../hybrid_performance_results.csv'


file1='commands_dijkstra.txt'
file2='graph_names.txt'
file3='../hybrid_originaldistance.txt'

make -f Makefile || exit


while read line1;
do
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'
echo 'FILE_TO_PROCESS : ' $line1
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'
while read line2;
do
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'
echo 'SIGNALS : ' $line2
echo '>>>>>>>>>>>>'
echo '>>>>>>>>>>>>'


if [ "$line2" == "1 0 0 0 0 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 0 1 0 3
done
./run $line2 $line1 0 0 1 1 3
rm 'time_results.txt'
fi


iter=$(head -n 1 $file3)


if [ "$line2" == "0 1 0 0 0 0 0 0 0" ];
then
for t in {1..16};
do
for n in {0..9};
do
./run $line2 $line1 $t $iter 1 0 3
done
./run $line2 $line1 $t $iter 1 1 3
rm 'time_results.txt'
done
fi


if [ "$line2" == "0 0 1 0 0 0 0 0 0" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 3
done
./run $line2 $line1 0 $iter 1 1 3
rm 'time_results.txt'
fi


if [ "$line2" == "0 0 0 1 0 0 0 0 0" ];
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


if [ "$line2" == "0 0 0 0 0 0 0 0 1" ];
then
for n in {0..9};
do
./run $line2 $line1 0 $iter 1 0 3
done
./run $line2 $line1 0 $iter 1 1 3
rm 'time_results.txt'
fi


if [ "$line2" == "0 1 0 1 0 0 0 0 0" ];
then
for i in {1..16};
do
for t in $(seq 0 $iter);
do
for n in {0..9};
do
./run $line2 $line1 $i $t 1 0 3
done
./run $line2 $line1 $i $t 1 1 3
rm 'time_results.txt'
done
done
fi


done < $file1
done < $file2
