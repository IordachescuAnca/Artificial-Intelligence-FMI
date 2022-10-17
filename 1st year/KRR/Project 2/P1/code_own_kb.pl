delete_neg([], []).
delete_neg([n(H)|T], [H|T1]):- delete_neg(T, T1), !.

append([], X, X).
append([H|T], X, [H|T1]) :-
    append(T, X, T1).

number_pos_literals([], 0).
number_pos_literals([n(_)|T], C) :- number_pos_literals(T, C),!.
number_pos_literals([_|T], C) :- number_pos_literals(T, Cnew), C is Cnew+1.

get_pos_literal([],0).
get_pos_literal([n(_)|T], Lit) :- get_pos_literal(T,Lit),!.
get_pos_literal([H|_], H).

backward_chaining_f([], _, 'YES').
backward_chaining_f([Goal|Goals], KB, Ans) :- member(Cl, KB), member(Goal, Cl), 
    delete(Cl, Goal, Cl1), delete_neg(Cl1, NewCl),append(Goals, NewCl, NewGoals), backward_chaining_f(NewGoals, KB, Ans), !.
backward_chaining_f(_, _, 'NO') :-!.

delete1(Val, [Val|X], X).
delete1(Val, [X|T1], [X|T2]):-
    delete1(Val, T1, T2).

perm([], []).
perm([X|T1], Ans):-perm(T1, Rez), delete1(X, Ans, Rez).


forward_chaining_f(Goals, _, S, 'YES') :- intersection(Goals, S, Int), perm(Int, Goals).
forward_chaining_f(Goals, KB,S, Ans) :- member(Cl, KB), member(Goal, Cl), not(Goal=n(_)), not(member(Goal, S)), delete(Cl, Goal, Cl1),
    									delete_neg(Cl1, NewCl), intersection(NewCl, S, S1), perm(S1, NewCl),
    					forward_chaining_f(Goals, KB, [Goal|S], Ans), !.
forward_chaining_f(_, _, _, 'NO') :- !.


days_q(Pred) :- 
  repeat, 
  writeln('How many days does the person intend to stay in a country? the answer is a number'),
  read(Days),nl,
  (Days > 90 -> 
    writeln('The person intends to stay in a country more than 90 days.'), Pred = n(not_stay_more_90_days), !
  ; writeln('The person does not intend to stay in a country more than 90 days.'), Pred = not_stay_more_90_days,
    ! 
  ).

age_q(Pred) :- 
  repeat, 
  writeln('How old is te person? the answer is a number'),
  read(Age),nl,
  (Age >= 14 -> 
    writeln('The person is at least 14 years old.'), Pred = age_14, !
  ; writeln('The person is at most 14 years old.'), Pred = n(age_14),
    !
  ).

purpose_q(Pred) :- 
  repeat, 
  writeln('Is the person purpose of the journey tourism/business?(the answer is yes/no)'),
  read(Choice),nl,
  (Choice == 'yes' -> 
    writeln('The purpose of the journey is tourism or business.'), Pred = jorney_t_b, !
  ; writeln('The purpose of the journey is not tourism or business,'), Pred = n(jorney_t_b),
    ! 
  ).


vaccinated_q(Pred) :- 
  repeat, 
  writeln('How many times has the person been vaccinated? the answer is a number'),
  read(Vacc),nl,
  (Vacc >= 2 -> 
    writeln('The person has been vaccinated at least 2 times.'), Pred = vacinnated_2_times, !
  ; writeln('The person has been vaccinated at most 2 times.'), Pred = n(vacinnated_2_times),
    !
  ).

questions([[P1],[P2],[P3],[P4]]) :- days_q(P1), age_q(P2), purpose_q(P3), vaccinated_q(P4).



main1(KB):- repeat, questions(X), 
		 append(KB, X, NewKB), backward_chaining_f([arrive_USA], NewKB, Ans), write('Backward chaining answer: '), writeln(Ans),
   		 forward_chaining_f([arrive_USA], NewKB,[], Ans1), write('Forward chaining answer: '), writeln(Ans1),
    	writeln('Write stop - stop the execution; Write anything else - continue.'), read(Choice), nl, (Choice == 'stop' -> 
    writeln('You selected to stop the execution of the program.'),!; writeln('You selected to continue.'), fail).



read_clauses(S,[]) :- at_end_of_stream(S).
read_clauses(S,[H|T]) :- not(at_end_of_stream(S)), read(S,H), read_clauses(S,T).

main:- open('own_kb.txt', read, S), read(S, H), close(S), main1(H).