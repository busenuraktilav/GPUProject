# GPUProject

######Graph generation code:

```
octave --persist kron_generator.m
kron_generator(scale,edge_factor,"path/to/example_graphs/graph_name.txt")
```


***scale*** can start from 18 and go to 26 (if memory allows it)
***edge_factor*** can be stable at 16
***graph_name*** should be as follow: graph_26_16_pos-10.txt



######Then, in CPU file, there is a script to run the program

```
bash runshell.sh
```
