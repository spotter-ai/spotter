(define (domain CUPS)
    (:requirements :strips :typing)
    (:types block cup - object)
    (:predicates (on ?x - object ?y - object)
	       (ontable ?x - object)
	       (clear ?x - object)
	       (handempty)
	       (holding ?x - object)
	       (inside ?x - block ?y - cup)
	       (facedown ?x - cup)
	       (upright ?x - cup)
	       (empty ?x - cup)
	       )
    (:action pickup
	     :parameters (?x - object)
	     :precondition (and (clear ?x) (ontable ?x) (handempty))
	     :effect
	     (and (not (ontable ?x))
		   (not (clear ?x))
		   (not (handempty))
		   (holding ?x)))
    (:action putdown
	     :parameters (?x - object)
	     :precondition (holding ?x)
	     :effect
	     (and (not (holding ?x))
		   (clear ?x)
		   (handempty)
		   (ontable ?x)))
  (:action stack
	     :parameters (?x - object ?y - object)
	     :precondition (and (holding ?x) (clear ?y))
	     :effect
	     (and (not (holding ?x))
		   (not (clear ?y))
		   (clear ?x)
		   (handempty)
		   (on ?x ?y)))
  (:action unstack
	     :parameters (?x - object ?y - object)
	     :precondition (and (on ?x ?y) (clear ?x) (handempty))
	     :effect
	     (and (holding ?x)
		   (clear ?y)
		   (not (clear ?x))
		   (not (handempty))
		   (not (on ?x ?y))))
;  (:action cover
;         :parameters (?c - cup ?b - block)
;         :precondition (and (holding ?c) (facedown ?c) (ontable ?b) (clear ?b) )
;         :effect
;         (and (handempty)
;            (not (holding ?c))
;            (inside ?b ?c)
;            (ontable ?c)
;            (not (clear ?b))))
;  (:action dropin
;         :parameters (?b - block ?c - cup)
;         :precondition (and (holding ?b) (upright ?c) (empty ?c))
;         :effect
;         (and (handempty)
;            (not (holding ?b))
;            (inside ?b ?c)))
;  (:action flipup
;         :parameters (?c - cup)
;         :precondition (and (facedown ?c) (holding ?c))
;         :effect
;         (and (upright ?c)
;            (not (facedown ?c))
;            (clear ?c)
;            (handempty)
;            (ontable ?c)
;            (not (holding ?c))))
)