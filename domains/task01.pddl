(define (problem CUPS-0-1)
(:domain CUPS)
(:objects B R - block C - cup)
(:INIT (CLEAR C) (CLEAR B) (CLEAR R) (ONTABLE C) (ONTABLE B)
 (ONTABLE R) (FACEDOWN C) (EMPTY C) (HANDEMPTY))
(:goal (AND (INSIDE B C) (UPRIGHT C)))
)