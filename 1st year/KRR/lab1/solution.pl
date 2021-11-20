f_max(X, Y, V):- X=<Y, V is Y.
f_max(X, Y, V):- X>Y, V is X.

f_member([H|_], H).
f_member([_|Tail], X) :- f_member(Tail, X).

f_concat([], L2, L2).
f_concat([H|Tail], L2, [H|L]) :- f_concat(Tail, L2, L).

f_alternate_sum_aux([], 0, _).
f_alternate_sum_aux([H|Tail], S, I) :- 0 is mod(I,2), Inew is I+1, f_alternate_sum_aux(Tail, Snew, Inew), S is Snew - H.
f_alternate_sum_aux([H|Tail], S, I) :- 1 is mod(I,2), Inew is I+1, f_alternate_sum_aux(Tail, Snew, Inew), S is Snew + H.
f_alternate_sum(List, S) :- f_alternate_sum_aux(List, S, 1).

f_eliminate([],_,[]).
f_eliminate([H|Tail], X, L) :- H==X, f_eliminate(Tail, X, L).
f_eliminate([H|Tail], X, [H|L]) :- H\=X, f_eliminate(Tail, X, L).

f_reverse([],X,X).
f_reverse([Head|T], L, Aux) :- f_reverse(T, L, [Head|Aux]).

f_all_permutations([], []).
f_all_permutations([Head|T1], [Head|T2]) :- f_all_permutations(T1, T2).
f_all_permutations([_|T1], T2) :- f_all_permutations(T1, T2).


f_occurence(_,[],0).
f_occurence(X,[X|T], S) :- f_occurence(X, T, Snew), S is Snew+1.
f_occurence(X,[_|T], S) :- f_occurence(X,T,S).

insertAt(X,1,L,[X|L]).
insertAt(X, Pos, [H|Tail], [H|Tail2]) :- PosNew is Pos-1, insertAt(X, PosNew, Tail, Tail2).

f_merge([],L1,L1).
f_merge(L1,[],L1).
f_merge([Head1|T1], [Head2|T2], [Head1|Z]) :- Head1=<Head2, f_merge(T1,[Head2|T2], Z).
f_merge([Head1|T1], [Head2|T2], [Head2|Z]) :- Head1>Head2, f_merge([Head1|T1],T2, Z).

