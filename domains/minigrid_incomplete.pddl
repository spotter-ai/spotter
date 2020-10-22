(define (domain gridworld_abstract)

  (:requirements :strips :typing )

  (:types thing - object
          agent physobject - thing
          notgraspable graspable - physobject
          wall floor door goal lava - notgraspable
          key ball box - graspable)

  (:predicates
           (nexttofacing ?a - agent ?thing - thing)
           (open ?t - door)
           (closed ?t - door)
           (locked ?t - door)
           (holding ?a - agent ?t - graspable)
           (handsfree ?a - agent)
           (obstructed ?a - agent)
           (blocked ?t - physobject)
           )

  ;(:action pickup
  ; :parameters (?a - agent ?thing - graspable)
  ; :precondition (and (handsfree ?a) (nexttofacing ?a ?thing) (not (holding ?a ?thing)))
  ; :unknown (and)
  ; :effect (and (not (handsfree ?a))
  ;              (not (nexttofacing ?a ?thing))
  ;              (not (obstructed ?a))
  ;              (holding ?a ?thing))
  ;)

  (:action putdown
   :parameters (?a - agent ?thing - graspable)
   :precondition (and (not (obstructed ?a)) (holding ?a ?thing))
   :unknown (and (blocked *))
   :effect (and (not (holding ?a ?thing))
                (handsfree ?a)
                (nexttofacing ?a ?thing)
                (obstructed ?a)
                ))

  (:action gotoobj1
   :parameters (?a - agent ?thing - graspable)
   :precondition (and (not (holding ?a ?thing)) (not (obstructed ?a)) (not (blocked ?thing)))
   :unknown (and)
   :effect (and (obstructed ?a)
                (nexttofacing ?a ?thing)))

  (:action gotoobj2
   :parameters (?a - agent ?thing - notgraspable)
   :precondition (and (not (obstructed ?a)) (not (blocked ?thing)))
   :unknown (and)
   :effect (and (obstructed ?a)
                (nexttofacing ?a ?thing)))

  (:action gotoobj3
   :parameters (?a - agent ?thing - graspable ?obstruction - thing)
   :precondition (and (not (holding ?a ?thing)) (nexttofacing ?a ?obstruction) (not (blocked ?thing)))
   :unknown (and)
   :effect (and (nexttofacing ?a ?thing)
                (not (nexttofacing ?a ?obstruction))))

  (:action gotoobj4
   :parameters (?a - agent ?thing - notgraspable ?obstruction - thing)
   :precondition (and (nexttofacing ?a ?obstruction) (not (blocked ?thing)))
   :unknown (and)
   :effect (and (nexttofacing ?a ?thing)
                (not (nexttofacing ?a ?obstruction))))

  (:action usekey
   :parameters (?a - agent ?key - key ?door - door)
   :precondition (and (nexttofacing ?a ?door) (holding ?a ?key) (locked ?door))
   :unknown (and)
   :effect (and (open ?door) (not (closed ?door)) (not (locked ?door))))
)
