class Simulador:
    def __init__(self):
        self.arrays=[
        [22,24,35,26,24,36,22,12,14,6,2,15,2,14,4,10,17,24,29,14,25,11,15,26,22,33,31,25,14,12,0,7,21,28,0,32,16,33,11,2,31,22,7,25,15,17,15,18,34,33,"salir"],
        [9,30,30,1,19,27,24,25,23,12,5,12,9,12,2,24,10,8,2,21,36,14,14,10,35,6,35,0,32,32,18,3,13,21,19,10,16,8,25,5,13,27,6,33,9,29,18,11,19,13,"salir"],
        [13,26,2,29,32,6,30,28,21,23,29,6,32,27,28,19,13,35,8,2,26,19,28,35,16,12,28,22,24,12,11,27,11,12,17,1,34,14,28,1,11,21,28,18,19,25,1,11,23,14,11,6,16,4,27,16,26,28,23,24,22,23,20,32,13,31,26,27,0,22,25,28,10,18,11,1,17,21,10,27,26,25,4,6,32,31,12,19,3,36,36,30,13,3,30,13,25,10,12,18,0,21,1,33,24,17,9,23,36,1,27,25,24,34,17,8,34,33,36,19,28,31,34,36,16,26,6,10,33,17,25,15,27,16,33,34,14,31,6,19,"salir"],
        [31,33,11,14,15,28,32,13,25,19,2,29,15,1,24,1,17,11,5,35,14,27,16,5,15,5,23,7,7,5,23,27,18,29,6,11,34,8,25,25,9,10,13,34,14,7,9,21,19,24,12,34,22,16,36,30,10,9,22,32,34,4,30,15,25,25,35,36,12,17,10,15,16,3,34,10,32,19,24,2,19,33,3,1,4,12,33,7,20,12,14,7,11,30,11,17,32,0,17,7,30,23,10,30,32,7,8,32,23,24,18,33,28,20,30,3,36,8,6,1,35,24,22,13,22,6,19,18,35,36,"salir"],
        [22,18,26,18,19,20,0,28,17,5,18,12,10,35,2,8,8,27,18,28,26,17,12,1,36,19,23,21,3,7,26,27,28,3,36,0,33,31,29,33,8,12,10,13,31,3,21,35,2,23,28,4,18,34,12,13,1,8,16,27,18,14,2,5,13,2,24,36,19,8,24,25,35,21,19,16,21,31,33,31,18,6,3,31,36,5,29,15,9,25,27,35,14,36,5,16,31,21,15,29,27,17,33,20,17,27,20,11,4,23,1,10,26,6,21,17,33,2,34,3,26,15,34,28,13,16,5,23,"salir"],
        [7,27,33,34,4,23,30,10,17,29,8,35,21,36,18,34,18,33,27,15,33,23,19,17,22,25,11,18,19,31,35,27,35,2,4,5,28,29,20,33,5,27,35,6,24,4,28,36,35,31,23,5,13,28,30,4,1,21,23,19,27,20,36,4,12,35,28,2,28,36,9,17,26,14,6,27,20,3,12,11,14,10,12,29,23,32,24,27,17,23,13,30,30,9,32,36,15,26,35,31,7,15,10,16,36,7,36,24,16,36,31,5,8,7,0,18,18,24,3,19,"salir"],
        [20, 33, 23, 10, 10, 4, 32, 25, 4, 32, 3, 7, 16, 9, 22, 8, 34, 35, 25, 30, 30, 26, 18, 28, 1, 19, 6, 32, 7, 31, 26, 30, 24, 1, 26, 4, 1, 24, 23, 6, 10, 0, 17, 9, 8, 12, 12, 19, 10, 35, 1, 20, 30, 16, 13, 26, 0, 7, 10, 34, 14, 16, 34, 18, 2, 30, 12, 24, 36, 17, 4, 32, 9, 20, 8, 35, 17, 20, 33, 0, 21, 10, 27, 4, 15,"salir"], 
        [8, 30, 24, 2, 5, 17, 36, 20, 21, 7, 6, 22, 12, 35, 24, 20, 4, 28, 7, 2, 24, 22, 27, 5, 21, 33, 12, 25, 3, 15, 12, 16, 17, 1, 24, 4, 9, 16, 27, 26, 15, 34, 21, 19, 25, 9, 3, 30, 17,"salir"], 
        [21, 36, 17, 22, 22, 13, 34, 10, 18, 29, 17, 17, 20, 28, 13, 35, 36, 23, 12, 12, 24, 29, 33, 1, 3, 31, 14, 18, 27, 32, 17, 5, 1, 27, 18, 36, 14, 35, 15, 6, 6, 15, 5, 0, 27, 3, 12, 6, 7, 11, 1, 27, 24, 19, 5, 12, 3,"salir"], 
        [32, 16, 23, 21, 11, 36, 26, 5, 3, 35, 28, 8, 11, 20, 27, 21, 23, 10, 13, 4, 0, 21, 27, 10, 21, 8, 11, 4, 35, 31, 23, 18, 0, 17, 1, 1, 9, 18, 25, 14, 29, 32, 12, 31, 26, 27, 2, 7, 21, 27, 36, 11, 12, 12, 22, 35, 13, 8, 10, 28,"salir"],
        [10, 12, 13, 34, 9, 25, 2, 7, 11, 21, 17, 34, 3, 8, 30, 2, 19, 22, 32, 20, 15, 20, 2, 3, 29, 16, 36, 18, 20, 29, 3, 36, 1, 6, 9, 18, 20, 15, 2, 27, 34, 35, 28, 13, 35, 23, 5, 35, 10, 22, 12, 28, 19, 26, 20, 10, 11, 27, 8, 0,"salir"],
        [36, 33, 27, 12, 16, 13, 0, 7, 20, 9, 36, 20, 30, 20, 20, 34, 31, 1, 18, 28, 9, 11, 26, 10, 9, 4, 4, 36, 35, 28, 8, 32, 8, 25, 5, 35,"salir"],
        [9, 21, 27, 4, 34, 23, 25, 32, 14, 27, 6, 32, 29, 6, 24, 18, 30, 1, 1, 10, 19, 7, 2, 12, 32, 11, 16, 36, 34, 6, 18, 0, 2, 23, 32, 8, 21, 6, 4, 16, 20, 3, 8, 16, 4, 1, 2, 18, 9, 28, 15, 30, 23, 6, 25, 33, 25, 32, 3, 29, 13, 22, 21, 20, 17, 17, 32, 22, 21, 26, 24, 21, 25, 15, 17, 17, 22, 7, 19, 24, 28, 20, 36, 12, 28, 5, 5, 27, 1, 16, 24, 10, 24, 15, 21, 20, 9,"salir"],
        [33, 30, 16, 14, 33, 28, 15, 12, 18, 16, 13, 24, 5, 15, 16, 10, 7, 27, 12, 16, 34, 9, 30, 5, 11, 24, 19, 19, 36, 25, 4, 7, 8, 6, 36, 19, 10, 33, 22, 29, 1, 35, 20, 24, 12, 3, 18, 10, 17, 0, 0, 20, 3, 36, 8, 6, 20, 36, 22, 5, 6, 27, 29, 29, 24, 14, 11, 13, 17, 14, 20, 32, 14, 5, 25, 18, 8, 13, 19, 0, 25, 5, 22, 10, 18, 33, 35, 14, 7, 11, 4, 28, 4, 29, 10, 31, 12, 0, 11, 19, 15, 21, 18, 30, 11, 15, 8, 16, 19, 15, 21, 1, 17, 23, 18, 20, 24, 15, 23, 21, 30, 20, 20, 19, 24, 4, 28, 29, 22, 31, 21, 1, 27, 29, 9, 19, 13, 5, 27, 31, 17, 22,"salir"]
            ]