unique([],[]).
unique([H|T],[H|TU]) :- delete(T,H,TN), unique(TN,TU).

count([],_,0).
count([Lit|T],Lit,Cnt):- count(T,Lit,NewCnt), Cnt is NewCnt+1.
count([H|T],Lit,Cnt):- H\=Lit,count(T,Lit,Cnt).

f_max([], List, List).
f_max([(H,_)|T], Max, List):- H >  Max, f_max(T, H, List). 
f_max([(H,_)|T], Max, List):- H =< Max, f_max(T, Max, List).
f_max([(H,_)|T], List):- f_max(T, H, List).

f_min([], List, List).
f_min([(H,_)|T], Min, List):- H >  Min, f_min(T, Min, List). 
f_min([(H,_)|T], Min, List):- H =< Min, f_min(T, H, List).
f_min([(H,_)|T], List):- f_min(T, H, List).


get_literal([],[]).
get_literal([n(H)|L], [H|R]) :- get_literal(L, R),!.
get_literal([H|L], [H|R]) :- get_literal(L, R).


get_literal_set(S, X) :- flatten(S, Ans), get_literal(Ans, Z), unique(Z, X). 

get_occurence(_,[],0).
get_occurence(V,[V|Tail], C) :- get_occurence(V,Tail,Cnew), C is Cnew+1.
get_occurence(V,[_|Tail], C) :- get_occurence(V,Tail,C).


get_occurence_set(V,S,C) :- flatten(S, Aux), get_literal(Aux, Ans), count(Ans,V,C).


get_list_occurence([],_,[]).
get_list_occurence([H|Tail], S, Ans) :- get_occurence_set(H, S, Count), get_list_occurence(Tail,S,Ans1), union([(Count, H)], Ans1, Ans).

select(X,[(X,Literal)|_],Literal).
select(V,[_|Tail],X) :- select(V,Tail,X).

strategy1(S, X) :- get_literal_set(S, Literals), get_list_occurence(Literals, S, ListOcc), f_max(ListOcc, Max), select(Max, ListOcc, X).
strategy2(S, X) :- get_literal_set(S, Literals), get_list_occurence(Literals, S, ListOcc), f_min(ListOcc, Min), select(Min, ListOcc, X).


dot_part1([],_,[]).
dot_part1([List|Tail], n(V), [List|X]) :- not(member(V, List)), not(member(n(V), List)), dot_part1(Tail, V, X),!.
dot_part1([List|Tail], n(V), X) :- member(V, List), dot_part1(Tail, V, X),!.
dot_part1([List|Tail], n(V), X) :- member(n(V), List), dot_part1(Tail, V, X),!.
dot_part1([List|Tail], V, [List|X]) :- not(member(V, List)), not(member(n(V), List)), dot_part1(Tail, V, X),!.
dot_part1([List|Tail], V, X) :- member(V, List), dot_part1(Tail, V, X),!.
dot_part1([List|Tail], V, X) :- member(n(V), List), dot_part1(Tail, V, X).

dot_part2([],_,[]).
dot_part2([List|Tail], n(V), X) :- not(member(n(V), List)), member(V, List),delete(List, V, NewList),dot_part2(Tail,n(V),Ans), union([NewList],Ans,X),!.
dot_part2([List|Tail],n(V),X) :- member(n(V), List), dot_part2(Tail,n(V),X),!.
dot_part2([List|Tail],n(V),X) :- not(member(V, List)), dot_part2(Tail,n(V),X),!.
dot_part2([List|Tail], V, X) :- not(member(V, List)), member(n(V), List),delete(List, n(V), NewList),dot_part2(Tail,V,Ans), union([NewList],Ans,X),!.
dot_part2([List|Tail],V,X) :- member(V, List), dot_part2(Tail,V,X),!.
dot_part2([List|Tail],V,X) :- not(member(n(V), List)), dot_part2(Tail,V,X),!.


dot(S, V, Ans) :- dot_part1(S,V,P1), dot_part2(S,V,P2), union(P1,P2,Ans).



aj_dp_strategy1([], 'Yes',[]) :- !.
aj_dp_strategy1(S, 'No',_) :- member([], S), !.
aj_dp_strategy1(S, Ans,[(V, 'true')|H]) :- strategy1(S, V), dot(S, V, X), aj_dp_strategy1(X, Ans,H).
aj_dp_strategy1(S, Ans,[(V, 'false')|H]) :- strategy1(S, V), dot(S, n(V), X), aj_dp_strategy1(X, Ans,H).

dp_strategy1([], 'Yes',[]) :- !.
dp_strategy1(S, 'No',[]) :- member([], S), !.
dp_strategy1(S, 'Yes',L) :- aj_dp_strategy1(S, 'Yes',L), !.
dp_strategy1(S, 'No',[]) :- aj_dp_strategy1(S, 'No',_), !.


aj_dp_strategy2([], 'Yes',[]) :- !.
aj_dp_strategy2(S, 'No',_) :- member([], S), !.
aj_dp_strategy2(S, Ans,[(V, 'true')|H]) :- strategy2(S, V), dot(S, V, X), aj_dp_strategy2(X, Ans,H).
aj_dp_strategy2(S, Ans,[(V, 'false')|H]) :- strategy2(S, V), dot(S, n(V), X), aj_dp_strategy2(X, Ans,H).

dp_strategy2([], 'Yes',[]) :- !.
dp_strategy2(S, 'No',[]) :- member([], S), !.
dp_strategy2(S, 'Yes',L) :- aj_dp_strategy2(S, 'Yes',L), !.
dp_strategy2(S, 'No',[]) :- aj_dp_strategy2(S, 'No',_), !.



write_ans1(_,[]):-!.
write_ans1(S1,[H|T]):- dp_strategy1(H, 'Yes', L), write(S1,H), write(S1," yes "), write(S1,L), write(S1,"\n"), write_ans1(S1,T), !.
write_ans1(S1,[H|T]):- write(S1,H), write(S1," no "), write(S1,"\n"), write_ans1(S1,T), !.

write_ans2(_,[]):-!.
write_ans2(S1,[H|T]):- dp_strategy2(H, 'Yes', L), write(S1,H), write(S1," yes "), write(S1,L), write(S1,"\n"), write_ans2(S1,T), !.
write_ans2(S1,[H|T]):- write(S1,H), write(S1," no "), write(S1,"\n"), write_ans2(S1,T), !.

read_clauses(S,[]) :- at_end_of_stream(S).
read_clauses(S,[H|T]) :- not(at_end_of_stream(S)), read(S,H), read_clauses(S,T).

main1:- open('input2.txt', read, S), open('output2.txt',write,S1), read_clauses(S, L), write_ans1(S1, L), close(S), close(S1).
main2:- open('input2.txt', read, S), open('output3.txt',write,S1), read_clauses(S, L), write_ans2(S1, L), close(S), close(S1).