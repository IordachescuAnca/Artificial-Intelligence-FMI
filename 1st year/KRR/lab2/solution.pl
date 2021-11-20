gcd(0,X,X):-!. 
gcd(X,0,X):-!. 
gcd(A,B,C):-A=<B, D is B-A, gcd(D,A,C),!.
gcd(A,B,C):-D is A-B, gcd(D,B,C).

split([],_,[],[]). 
split([H1|T],H,[H1|A],B):-H1=<H,!, split(T,H,A,B).
split([H1|T],H,A,[H1|B]):-H1>H,!, split(T,H,A,B).

insertsort([],[]). 
insertsort([X|T],S):-insertsort(T,ST),insert(X,ST,S). 

insert(Elem,[Head|T1], [Head|T2]) :- Elem>Head, insert(Elem, T1, T2),!.
insert(Elem,L,[Elem|L]).

quick([],[]). 
quick([X],[X]). 
quick([H|T],L):-split(T,H,A,B),quick(A,A1),quick(B,B1), append(A1, [H|B1], L).

queen([]). 
queen([[X,Y]|S]):-queen(S),member(Y,[1,2,3,4,5,6,7,8]), not(atack([X,Y],S)). 
 
atack([X,_],S):-member([X1,_],S), X1==X,!.
atack([_,Y],S):-member([_,Y1],S), Y==Y1,!.
atack([X,Y],S):-member([X1,Y1],S), Val1 is abs(X-X1), Val2 is abs(Y-Y1), Val1==Val2.



la_dreapta(X,Y) :- X =:= Y + 1.
 
la_stanga(X,Y) :- X =:= Y - 1.
 
langa(X, Y) :- la_dreapta(X,Y).
langa(X, Y) :- la_stanga(X,Y).
 
 
solutie(Sol,FishOwner) :- Sol = [
casa(1,_,_,_,_,_),
casa(2,_,_,_,_,_),
casa(3,_,_,_,_,_),
casa(4,_,_,_,_,_),
casa(5,_,_,_,_,_)],
 
member(casa(_,british,red,_,_,_), Sol),
 
member(casa(S,_,blue,_,_,_), Sol),
member(casa(P,norwegian,_,_,_,_), Sol),
S =:= P + 1,
    
 
member(casa(A,_,green,_,_,_), Sol),
member(casa(B,_,white,_,_,_), Sol),
la_stanga(A, B),
 
 
member(casa(_,_,green,_,coffee,_), Sol),
 
member(casa(3,_,_,_,milk,_), Sol),
 
member(casa(_,_,yellow,_,_,'Dunhill'), Sol),
 
member(casa(1,norwegian,_,_,_,_), Sol),
 
member(casa(_,swedish,_,dog,_,_), Sol),
 
member(casa(_,_,_,bird,_,'Pall Mall'), Sol),
 
member(casa(Z,_,_,cat,_,_), Sol),
member(casa(T,_,_,_,_,'Malboro'), Sol),
langa(Z, T),
 
member(casa(_,_,_,_,beer,'Winfield'), Sol),
 
member(casa(U,_,_,horse,_,_), Sol),
member(casa(V,_,_,_,_,'Dunhill'), Sol),
langa(U, V),
 
member(casa(_,german,_,_,_,'Rothmans'), Sol),
 
member(casa(X,_,_,_,_,'Malboro'), Sol),
member(casa(Y,_,_,_,water,_), Sol),
langa(Y, X),
 
member(casa(_,FishOwner,_,fish,_,_), Sol).