:-dynamic res/1.

get_clause_pos([],_,[]).
get_clause_pos([List|_],Literal,List) :- member(Literal,List).
get_clause_pos([_|R],Literal,Cl) :- get_clause_pos(R,Literal,Cl).

get_clause_neg([],_,[]).
get_clause_neg([List|_],Literal,List) :- member(n(Literal),List).
get_clause_neg([_|R],Literal,Cl) :- get_clause_neg(R,Literal,Cl).

get_solvent(X, [n(X)],X,[]).
get_solvent(List1, List2, Literal, X) :- delete(List1, Literal, L1), delete(List2, n(Literal), L2), union(L1,L2,X).



unique([],[]).
unique([H|T],[H|TU]) :- delete(T,H,TN), unique(TN,TU).

get_literal([],[]).
get_literal([n(H)|L], [H|R]) :- get_literal(L, R),!.
get_literal([H|L], [H|R]) :- get_literal(L, R).

get_literal_set(S, X) :- flatten(S, Ans), get_literal(Ans, Z), unique(Z, X).

res(KB, 'Unsat') :- member([],KB),!.

res(KB, Y) :- get_literal_set(KB, Literals), member(X, Literals), get_clause_pos(KB,X,LP), get_clause_neg(KB,X,LN),
    LP \= [], LN \= [], get_solvent(LP, LN, X, Sol), not(member(Sol, KB)), union(KB, [Sol], RS), res(RS, Y),!.
res(KB, 'Sat') :- get_literal_set(KB, Literals), member(X, Literals), get_clause_pos(KB,X,LP), get_clause_neg(KB,X,LN),
    LP \= [], LN \= [], get_solvent(LP, LN, X, Sol), member(Sol, KB),!.

res(KB, 'Sat') :- get_literal_set(KB, Literals), member(X, Literals), get_clause_pos(KB,X,LP),
    LP == [].
res(KB, 'Sat') :- get_literal_set(KB, Literals), member(X, Literals), get_clause_neg(KB,X,LN),
    LN == [].

read_clauses(S,[]) :- at_end_of_stream(S).
read_clauses(S,[H|T]) :- not(at_end_of_stream(S)), read(S,H), read_clauses(S,T).

write_ans(_,[]):-!.
write_ans(S1,[H|T]):- res(H, A), write(S1,H), write(S1," "), write(S1,A), write(S1,"\n"), write_ans(S1,T), !.

main:- open('input.txt', read, S), open('output.txt', write, S1), read_clauses(S, L), write_ans(S1,L), close(S), close(S1).
