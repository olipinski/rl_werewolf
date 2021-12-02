## Available data

This dir contains code and files which deals with the understanding on what the agents are learning during training.

### Target matrix [TM]

The good news is that we only have to focus on the target matrix. This matrix is square and symmetric, in which every
row is the agent preference list for target and the column is just an index of preference. Element _Tij_ is an integer
in range `[0, np]` representing the j-th preference for agent i-th. The lower the number _j_ the higher the agent _i_
wants him dead. Rows are not unique.

Example [TM]:

| agents | v1  | v2  | v3  | v4  | v5  |
|--------|-----|-----|-----|-----|-----|
| ag0    | 4   | 1   | 3   | 3   | 0   |
| ag1    | 1   | 0   | 0   | 4   | 0   |
| ag2    | 3   | 0   | 2   | 0   | 3   |
| ag3    | 3   | 4   | 0   | 1   | 1   |
| ag4    | 0   | 0   | 3   | 1   | 0   |

#### Stacked agent targets [SAT]

Since an episode in made of multiple target matrices, we can combine rows from each agent to build a stacked target
matrix.

Each row would be a different vote from the same agent on a different day.

Example:

| day | v1  | v2  | v3  | v4  | v5  |
|-----|-----|-----|-----|-----|-----|
| d1  | 2   | 0   | 1   | 4   | 3   |
| d2  | 1   | 3   | 2   | 1   | 2   |
| d3  | 0   | 0   | 1   | 3   | 1   |
| d4  | 1   | 1   | 1   | 3   | 1   |
| d5  | 3   | 3   | 3   | 3   | 3   |

This can be used for some evaluation metrics

##### Decreasing number of possibilities

Thanks to the introduction of the parametric action environment each agent is limited to choose alive agents for their
target outputs. This means that with each day the number of unique elements in a row decreases by one. This must be
taken into account for some evaluation metrics.

#### Target Difference

The `trg_diff` metric is simply the number of different agent ids between vote time and execution time. Given the output
vector for each agent (of size `[num players, num players]`) the target difference is estimated as follows:

    diff = np.sum(cur_targets != prev_targets) / (num_player ** 2)

Where `prev_targets ` is the vote output and `cur_targets` is the execution one.

The aim of this metric is to check the consistency of vote/execution during the training.

# Metrics

In this section possible ideas for evaluation will be stored.

## Role invariant

This class of metrics does not depend on the roles each agent has.

### Distance functions

A distance function takes as input two vectors and return real number, possibly in range `[0,1]`.

#### Index distance function [IDF]

For evaluating some kind of distance function must be defined. The Index Distance Function [IDF] is one of these, its
purpose is to estimate how much of a difference there is between to consecutive votes.

Having two consecutive target vectors $r_j$, $r_{j+1}$ of size `n`; we have:

$IDF(r_j,r_{j+1})={\frac{\sum_{i=1}^n|indexof(r_{j,i},r_{j+1})-i|}{n-1}}$

Where the `indexof(elem,list)` is a function which return the index of the first occurrence of `elem` in `list`. The
divisor guarantees that we are returned a number in range `[0,1]`.

The idea for this distance function is to measure how much a certain vote has been discarded/upgraded from the previous
position.

#### Repetition distance function [RDF]

Close to IDF, this distance function counts the times a certain vote is repeated in to consecutive outputs and estimate
the normalized difference.

Given two rows ($r_j$, $r_{j+1}$), and a list of unique value in $r_1$ of size $n$:

$RDF(r_j,r_{j+1})=\frac{\sum_{i=1}^n{|freq(r_{j,i},r_j)-freq(r_{j+1,i},r_{j+1})|}}{n}$

Where the `freq(elem, list)` function counts the repetition `elem` in a list.

#### Set difference function [SDF]

Given to row vectors $r_1,r_2$ of size $n$ the function returns the number of different elements. Example:

- $r_1=[1,2,3,2,1,4,5]$
- $r_2=[5,4,2,3,3,1,4]$
- $r_1-r_2=[2,1]$
- $r_2-r_1=[3,4]$

$SDF(r_1,r_2)=length(r_1-r_2)$

### Matrix distance functions

This class of functions takes as input a matrix and returns a scalar, possibly in range `[0,1]`.

#### Difference of SAT [DSAT]

We can evaluate the difference between consecutive votes for each agent. Taking into account the STTM matrix for some
agent we retrieve two consecutive rows ($r_i$ and $r_{i+1}$). For the comparison to be fair we need to save the number
of alive agent at each day in a vector which will be called `alive`.

Trimming both the rows and getting the first $alive_i$ elements will guarantee a fair comparison. The formula is as
follows:

$DSAT =\sum_{i=1}^{days}{IDF(r_i,r_{i+1},alive_i)}/days$

#### Modified DSAT [MDSAT]

We can update the DSAT further by adding the RDF to its formula.

$MDSAT =\sum_{i=1}^{days}{\frac{IDF(r_i,r_{i+1},alive_i)+RDF(r_i,r_{i+1})}{2* days}}$

#### General accordance [GA]

This distance function takes as input the TM matrix and returns a scalar in range `[0,1]` measuring the accordance
between agents. Ideally a value equal to one means that each agent outputs the same target vector, while zero *******.

It is split in two parts:

1. First it sums the number of unique values columns wise normalizing it by `np`.
2. Then it compares each row with all the others, summing the number of different values. Then normalize the result by
   $2*np^2$

### Other distance algorithms

#### Target Influence function [TIF]

This metric tries to measure how much each individual agent is influenced by the general vote phase.

The aim of this metric is seeing how much the TM at communication time influences the one at execution time. Seeing this
value increasing means that agents are taking into high consideration past votes for their next one. On the other hand,
when the value decreases then the agents may make use of the voting phase as something completely different from what
was intended for, further metrics should be created on this subject.

The algorithm takes as input the communication row for agent _i_ [$cm_i$], the execution row [$er_i$] and the
communication TM [$CTM$]. It works column wise rather than row wise since the preference for a certain agent is
proportional to its index in the output vector. Further implementation could consider the row wise counterpart.

##### Scalar influence function [SIF]

To keep the algorithm simple we first consider the case in which the TM has size $[np,1]$, that is every agent outputs
just one vote.

In this case we have two scalars for agent _i_ (communication/execution outputs [$e_i$]/[$c_i$])  and a column
vector [$cm$] which is the output of every agent, notice that we remove $c_i$ from $cm$ in order to not count the
agent _i_ influence on itself.

If $c_i == e_i$ then $SIF=0$, the agent was apparently not influenced by the communication phase.

Else the formula is as follows:

$SIF(c_i,e_i,cm)=\frac{freq(e_i,cm)}{np-1}$

Where $np$ is the number of players.

From now on the TIF algorithm is easy to understand. For each column in the CTM (where there is no $cm_i$ anymore) you
compute the SIF. Finally, you normalize dividing by the number of players $np$.

## Role dependent

something

## General ideas

Just some random ideas which requires further implementations:

- diversity between night and day communication for wolves
- see if villagers who vote for wolves more often are then targeted by the latter during nighttime.
- check if there is diversity between wolves and vil when comes to day communication time.
- check for regular pattern somehow, can be used to communicate.
- accordance in communication/execution phase between every agent + highlighting difference accordance in roles.