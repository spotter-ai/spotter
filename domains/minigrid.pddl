(define (domain gridworld_abstract)

  (:requirements :strips :typing )

  (:types thing room - object
          agent physobject - thing
          door nondoor - physobject
          notgraspable graspable - nondoor
          wall floor goal lava - notgraspable
          key ball box - graspable)

  (:predicates
           (nexttofacing ?a - agent ?thing - physobject)
           (open ?t - door)
           (closed ?t - door)
           (locked ?t - door)
           (holding ?a - agent ?t - graspable)
           (handsfree ?a - agent)
           (obstructed ?a - agent)
           (blocked ?t - door)
           (atgoal ?a - agent ?g - goal)
           (inroom ?a - agent ?g - physobject)
           )

  (:action pickup
   :parameters (?a - agent ?thing - graspable)
   :precondition (and (handsfree ?a) (nexttofacing ?a ?thing) (not (holding ?a ?thing)))
   :unknown (and)
   :effect (and (not (handsfree ?a))
                (not (nexttofacing ?a ?thing))
                (not (obstructed ?a))
                (holding ?a ?thing))
  )

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
   :precondition (and (not (holding ?a ?thing)) (not (obstructed ?a)) (inroom ?a ?thing))
   :unknown (and)
   :effect (and (obstructed ?a)
                (nexttofacing ?a ?thing)))

  (:action gotoobj2
   :parameters (?a - agent ?thing - notgraspable)
   :precondition (and (not (obstructed ?a)) (inroom ?a ?thing))
   :unknown (and)
   :effect (and (obstructed ?a)
                (nexttofacing ?a ?thing)))

  (:action gotoobj3
   :parameters (?a - agent ?thing - graspable ?obstruction - thing)
   :precondition (and (not (holding ?a ?thing)) (nexttofacing ?a ?obstruction) (inroom ?a ?thing))
   :unknown (and)
   :effect (and (nexttofacing ?a ?thing)
                (not (nexttofacing ?a ?obstruction))))

  (:action gotoobj4
   :parameters (?a - agent ?thing - notgraspable ?obstruction - thing)
   :precondition (and (nexttofacing ?a ?obstruction) (inroom ?a ?thing))
   :unknown (and)
   :effect (and (nexttofacing ?a ?thing)
                (not (nexttofacing ?a ?obstruction))))

  (:action usekey
   :parameters (?a - agent ?key - key ?door - door)
   :precondition (and (nexttofacing ?a ?door) (holding ?a ?key) (locked ?door))
   :unknown (and)
   :effect (and (open ?door) (not (closed ?door)) (not (locked ?door))))

  (:action opendoor
   :parameters (?a - agent ?door - door)
   :precondition (and (nexttofacing ?a ?door) (closed ?door))
   :unknown (and)
   :effect (and (open ?door) (not (closed ?door)) (not (locked ?door))))

  (:action stepinto
   :parameters (?a - agent ?g - goal)
   :precondition (and (nexttofacing ?a ?g))
   :unknown (and (nexttofacing ?a *))
   :effect (and (not (nexttofacing ?a ?g)) (atgoal ?a ?g)))

  (:action gotodoor1
   :parameters (?a - agent ?thing - door)
   :precondition (and (not (obstructed ?a)) (not (blocked ?thing)) (inroom ?a ?thing))
   :unknown (and)
   :effect (and (obstructed ?a)
                (nexttofacing ?a ?thing)))

  (:action gotodoor2
   :parameters (?a - agent ?thing - door ?obstruction - thing)
   :precondition (and (nexttofacing ?a ?obstruction) (not (blocked ?thing)) (inroom ?a ?thing))
   :unknown (and)
   :effect (and (nexttofacing ?a ?thing)
                (not (nexttofacing ?a ?obstruction))))

  (:action enterroomof
   :parameters (?a - agent ?d - door ?g - physobject)
   :precondition (and (not (blocked ?d)) (nexttofacing ?a ?d) (open ?d))
   :unknown (and)
   :effect (and (inroom ?a ?g) (not (nexttofacing ?a ?d)) (not (obstructed ?a))))

)
