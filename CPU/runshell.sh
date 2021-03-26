#!/bin/bash

file1='commands_bellman.txt'
file2='graph_names.txt'
file3='../bellman_originaldistance.txt'

make -f Makefile_bellman || exit

while read line1;
do
while read line2;
do
echo $line2

./run $line2 $line1

iter=$(head -n 1 $file3)
maxEdge=$(sed -n '2p' $file3)

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

./run $line2 $line1

iter=$(head -n 1 $file3)
maxEdge=$(sed -n '2p' $file3)

done < $file1
done < $file2


