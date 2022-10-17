remove_negations([], []):-!.
remove_negations([n(P)|Q], [P|R]):- remove_negations(Q, R), !.

append([], List, List).
append([Head|Tail], List, [Head|Rest]) :-
    append(Tail, List, Rest).

number_pos_literals([], 0).
number_pos_literals([n(_)|T], C) :- number_pos_literals(T, C),!.
number_pos_literals([_|T], C) :- number_pos_literals(T, Cnew), C is Cnew+1.

get_pos_literal([],0).
get_pos_literal([n(_)|T], Lit) :- get_pos_literal(T,Lit),!.
get_pos_literal([H|_], H).

backward_chaining_f([], _, 'YES').
backward_chaining_f([Goal|Goals], KB, Ans) :- member(Cl, KB), member(Goal, Cl), 
    delete(Cl, Goal, Cl1), remove_negations(Cl1, NewCl),append(Goals, NewCl, NewGoals), backward_chaining_f(NewGoals, KB, Ans), !.
backward_chaining_f(_, _, 'NO') :-!.

delete1(X, [X|T], T).
delete1(X, [H|T], [H|S]):-
    delete1(X, T, S).

permutation([], []).
permutation([H|T], R):-
    permutation(T, X), delete1(H, R, X).


forward_chaining_f(Goals, _, S, 'YES') :- intersection(Goals, S, Int), permutation(Int, Goals).
forward_chaining_f(Goals, KB,S, Ans) :- member(Cl, KB), member(Goal, Cl), not(Goal=n(_)), not(member(Goal, S)), delete(Cl, Goal, Cl1),
    									remove_negations(Cl1, NewCl), intersection(NewCl, S, S1), permutation(S1, NewCl),
    					forward_chaining_f(Goals, KB, [Goal|S], Ans), !.
forward_chaining_f(_, _, _, 'NO') :- !.


temperature_q(Pred) :- 
  repeat, 
  writeln('What is patient temperature? the answer is a number'),
  read(Temp),nl,
  (Temp > 38 -> 
    writeln('The temperature is higher than 38.'), Pred = high_temp, !
  ; writeln('The temperature is lower than 38.'), Pred = n(high_temp),
    ! 
  ).

sick_days_q(Pred) :- 
  repeat, 
  writeln('For how many days has the patient been sick? the answer is a number'),
  read(Days),nl,
  (Days >= 2 -> 
    writeln('Patient was sick for at least 2 days.'), Pred = sick_2_days, !
  ; writeln('Patient was sick for at most 2 days.'), Pred = n(sick_2_days),
    !
  ).

muscle_pain_q(Pred) :- 
  repeat, 
  writeln('Has patient muscle pain? (the answer is yes/no)'),
  read(Pain),nl,
  (Pain == 'yes' -> 
    writeln('Patient has muscle pain.'), Pred = muscle_pain, !
  ; writeln('Patient does not have muscle pain.'), Pred = n(muscle_pain),
    ! 
  ).


cough_q(Pred) :- 
  repeat, 
  writeln('Has patient cough? (the answer is yes/no)'),
  read(Cough),nl,
  (Cough == 'yes' -> 
    writeln('Patient has cough.'), Pred = cough, !
  ; writeln('Patient has not cough.'), Pred = n(cough),
    ! 
  ).

questions([[P1],[P2],[P3],[P4]]) :- temperature_q(P1), sick_days_q(P2), muscle_pain_q(P3), cough_q(P4).



main1(KB):- repeat, questions(X), 
		 append(KB, X, NewKB), backward_chaining_f([pneumonia], NewKB, Ans), write('Backward chaining answer: '), writeln(Ans),
   		 forward_chaining_f([pneumonia], NewKB,[], Ans1), write('Forward chaining answer: '), writeln(Ans1),
    	writeln('Write stop - stop the execution; Write anything else - continue.'), read(Choice), nl, (Choice == 'stop' -> 
    writeln('You selected to stop the execution of the program.'),!; writeln('You selected to continue.'), fail).


read_clauses(S,[]) :- at_end_of_stream(S).
read_clauses(S,[H|T]) :- not(at_end_of_stream(S)), read(S,H), read_clauses(S,T).

main:- open('course_kb.txt', read, S), read(S, H), close(S), main1(H).