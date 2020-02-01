#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <ctime>
#include <fstream>
#include <time.h>
// #include <cuda.h>

using namespace std;

#define N            3
#define BOARD_WIDTH  9
#define BOARD_SIZE   81   //3: 81, 4: 256, 5: 625, 6: 1296, 7: 2401
#define DIFF         8   //3:  8, 4: 14,  5:  22, 6:   32, 7:   44
#define N_ASSIGN     17    //3: 17, 4: 30,  5:  47, 6:   68, 7:   93
#define SET_BIT      0x1FF   //9 = 0x1FF, 16 = 0xFFFF, 25 = 0x1FFFFFF, 36 = 0xFFFFFFFFF, 49 = 0x1FFFFFFFFFFFF, 64 = 0xFFFFFFFFFFFFFFFF

/*
 * Possible class represents each cell of the Sudoku board
 * It is just a 32bit(up to N=5) unsigned integer to save each possible number into a bit
 * bool is_on(int i)       : Check if number i is possible number for the cell
 * void eliminate(int i)   : Remove number i from possible number int the cell
 * int val()               : Returns the lowest possible number in the cell
 * int count()             : Count possible numbers in the cell
 * string str(int width)   : Print out assgined number of the cell
*/

class Possible {
   uint32_t _b;   //using bitwise 
public:
   Possible() : _b(SET_BIT) {}  
   Possible(uint32_t num) : _b(num) {}  
   bool   is_on(int i) const { return ((_b >> (i-1)) & 1); } 
   void   eliminate(int i)   { _b &= ~(uint32_t(1) << (i-1)); }   
   int    val()        const;
   int    count()      const;
   string str(int width) const;
};

/*
 * int val() : Returns the lowest possible number in the cell
*/
int Possible::val() const {
   // _b should not be 0, there should be at least one possible number
   if (_b == 0) return -1;

   uint32_t count = 0, n = _b;
   // shift bits until meet bit 1
   while ( (n & 1) == 0) {
      n >>= 1;
      ++count;
   }
   // 1 should be added because 0-th bit represent number 1
   return count + 1;
}

/*
 * int count() : Count possible numbers in the cell
 */
int Possible::count() const {
   uint32_t count = 0, n = _b;
   // count all 1 bits
   while (n > 0) {
      count += n & 1;
      n >>= 1;
   }
   return count;
}

/*
 * string str(int width)   : Print out assgined number of the cell
 */

string Possible::str(int width) const {
   string s(width, ' ');
   if (count() == 1) {
      int i = val();
      s = string(width - to_string(i).size(), ' ') + to_string(i);
   }
   return s;
} 

/*
 * Sudoku board is matrix of Possible cells
 * Possible possible(int k)             : Returns k-th cell information
 * bool is_solved() const               : Checks if the Sudoku board has been solved
 * bool assign(int k, int val)          : Assign val to cell k
 * void least_count(int* least_k) const : Returns the list of cell index of cell with least possible numbers left
 * void write(ostream& o) const         : Print the solved Sudoku board
 * int unassigned() const               : Checks number of unassigned Sudoku cells for logging purpose only
 * string input_string()                : Prints out string form of current Sudoku board. This is for problem generation.
 */

class Sudoku {
   static uint16_t _group[3*BOARD_WIDTH][BOARD_WIDTH], _neighbors[BOARD_SIZE][3*BOARD_WIDTH-3], _groups_of[BOARD_SIZE][3];
    Possible _cells[BOARD_SIZE];
   bool     eliminate(int k, int val);
public:
   Sudoku() { };
   Sudoku(string s);
   static void init();

   Possible possible(int k) const { return _cells[k]; }
   bool     is_solved() const;
   bool     assign(int k, int val);
   void     least_count(int* least_k) const;
   void     write(ostream& o) const;
   int      unassigned() const;
   string   input_string();
   friend void rand_generator(int num);
};


uint16_t
Sudoku::_group[3*BOARD_WIDTH][BOARD_WIDTH] = {}, Sudoku::_neighbors[BOARD_SIZE][3*BOARD_WIDTH-3]={}, Sudoku::_groups_of[BOARD_SIZE][3]={};
// initializing static valiables, which designates gruop numbers and neighbors of a cell
// _group : total 3 * BOARD_WIDTH, 1 for row group, 2 for column group, 1 for square group
// _groups_of[k] : what group does a _cell[k] belongs to. Each cell belong to 3 groups
// _neighbors[k] : neighbors of _cell[k] excluding itself
void Sudoku::init() {
   for (int i = 0; i < BOARD_WIDTH; ++i) {
      for (int j = 0; j < BOARD_WIDTH; ++j) {
         const int k = i*BOARD_WIDTH + j;
         //Which group does k belongs to, x[0]: row group, x[1]: column gorup, x[2]: square group
         const int x[3] = {i, BOARD_WIDTH + j, 2*BOARD_WIDTH + (i/N)*N + j/N};

         _group[x[0]][j] = k;
         _group[x[1]][i] = k;
         _group[x[2]][(i%N)*N + (j%N)] = k;

         for (int g = 0; g < 3; g++) 
            _groups_of[k][g] = x[g];
      }
   }

   for (int k = 0; k < BOARD_SIZE; ++k) {
      int l = 0;
      for (int x = 0; x < 3; x++) {
         for (int j = 0; j < BOARD_WIDTH; ++j) {
            int k2 = _group[_groups_of[k][x]][j];
            if (k2 != k) _neighbors[k][l++] = k2;
         }
      }
   }
}

/* 
 * string input_string(): Prints out string from generated Sudoku problems
 */
string Sudoku::input_string() {
   string cell_val[BOARD_SIZE];
   for (int i = 0; i < BOARD_SIZE; ++i) {
      if (_cells[i].count() == 1)
         cell_val[i] = ' ' + to_string(_cells[i].val()) + ' ';
      else 
         cell_val[i] = " 0 ";
   }

   string ret = "";
   for (int i = 0; i < BOARD_SIZE; ++i) {
      ret += cell_val[i];
   }
   ret += '\n';
   return ret;
}

/*
 * int Sudoku::unassigned() : Count number of all the unassigned Sudoku cell.
 */
int Sudoku::unassigned() const {
   int count = 0;
   for (int k = 0; k < BOARD_SIZE; ++k) 
      //Cells with an assigned number will have count = 1 
      if (_cells[k].count() != 1) 
         ++count;
   return count;
}
/*
 * bool Sudoku::is_solved() const : Check if Sudoku board is solved by checking count of all cells are 1
 */
bool Sudoku::is_solved() const {
   for (int k = 0; k < BOARD_SIZE; ++k) 
      if (_cells[k].count() != 1) 
         return false;
   return true;
}
/*
 * void write(ostream& o) const : Print the solved Sudoku board
 */
void Sudoku::write(ostream& o) const {
   int width = to_string(BOARD_WIDTH).size() + 1;
   for (int k = 0; k < BOARD_SIZE; ++k) {
      width = max(width, 1 + _cells[k].count());
   }
   width = 3;

   const string sep(N * width + 1, '-');
   for (int i = 0; i < BOARD_WIDTH; ++i) {
      if (i > 0 && i % N == 0) {
         for (int k = 0; k < N - 1; ++k)
            o << sep << '+';
         o << sep << endl;
      }
      for (int j = 0; j < BOARD_WIDTH; ++j) {
         if (j > 0 && j % N == 0) 
            o << " |";
         o << _cells[i*BOARD_WIDTH + j].str(width);
      }
      o << endl;
   }
   o << endl;
}

/*
 * bool Sudoku::assign(int k, int val) : Assign val to cell k
 * Assigning a value is eliminating all other possible numbers from cell k
 * assign and eliminate will do the constant propagation back and forth
 */
bool Sudoku::assign(int k, int val) {
   for (int i = 1; i <= BOARD_WIDTH; ++i) {
      if (i != val) {
         //eliminate other values, if not possible wrong approach
         if (!eliminate(k, i)) return false;
      }
   }
   return true;
}

/*
 * bool Sudoku::eliminate(int k, int val) : Eliminate val from cell k and do the constant propagation
 */
bool Sudoku::eliminate(int k, int val) {
   // if val is already eliminated in _cell[k] return
   if (!_cells[k].is_on(val)) {
      return true;
   }

   _cells[k].eliminate(val);

   // count number of all posible values left for cell[k] after removing val
   const int n = _cells[k].count();
   // N should be at least 1, it is a conflict if n is 0
   if (n == 0) {
      return false;
   // if there is only one possible value left for cell[k] assigned that value to the _cell[k]
   // and eliminate that number from neighbors 
   } else if (n == 1) {
      // find that unique value v
      const int v = _cells[k].val();
      // remove v from all the neighbors
      for (int i = 0; i < 3*BOARD_WIDTH-3; ++i) {
         if (!eliminate(_neighbors[k][i], v)) return false;
      }
   }

   // Check count of possible numbers of the neighbors to check conflict or assign a number
   // there are three possible group for each cell[k]
   for (int i = 0; i < 3; ++i) {
      // get the group number x
      const int x = _groups_of[k][i];
      int n = 0, ks;
      // for each neighbors in the group x
      for (int j = 0; j < BOARD_WIDTH; ++j) {
         const int p = _group[x][j];
         // count the number of possible values for each neighbor in the i-th group
         if (_cells[p].is_on(val)) {
            n++, ks = p;
         }
      }
      // it's a conflict if n = 0 for val in the group
      if (n == 0) {
         return false;
      // if there is only one possible value, assign such value. Constraint propagation
      } else if (n == 1) {
         if (!assign(ks, val)) {
            return false;
         }
      }
   }
   return true;
}

/*
 * void Sudoku::least_count(int* least_k) const : Returns array of cell number k with least possible numbers left
 * It returns am array to check if there is non-unique solution but only the first element of array is used
 */
void Sudoku::least_count(int* least_k) const {
   int k = -1, min;

   for (int i = 0; i < BOARD_SIZE; ++i) {
      // get the count of possible numbers a cell can take
      const int m = _cells[i].count();
      // update minimum
      if (m > 1 && (k == -1 || m < min)) {
         min = m, k = i;
      }
   }

   int l = 0;
   for (int i = 0; i < BOARD_SIZE; ++i) {
      const int m = _cells[i].count();
      // add the cell number with least possible numbers left
      if (m == min) {
         least_k[l++] = i;
      }
   }
}
 /*
 *Sudoku::Sudoku(string s) : Parse Sudoku board from a string
 */
Sudoku::Sudoku(string s) {
   int k = 0;
   for (int i = 0; i < s.size(); ++i) {
      // if there is a single  mark it's 2 digit number
      if (s[i] == ' ') {
         int last = s.substr(i+1).find(' ');
         int num = stoi(s.substr(i+1, last));
         // assign number to cell k and check if there is any conflict
         if (num != 0 && !assign(k,num)) {
            cerr << "error" << endl;
            return;
         }
         ++k;
         i += last + 1;
      }
      else if (s[i] >= '1' && s[i] <= '9') {
         if (!assign(k, s[i] - '0')) {
            cerr << "error" << endl;
            return;
         }
         ++k;
      } 
        // 0 or . can be used to unassigned cells
      else if (s[i] == '0' || s[i] == '.') {
         ++k;
      }
   }
}

// This is to save number of attempt performed during each recursion depth
int attempt_per_depth[BOARD_SIZE];

/*
 * Sudoku* solve(Sudoku* S, int depth) : Solves Sudoku
 * Algorithm: From one of the cell with least possible numbers left
              choose one number and assign that number to the cell.
              If it was a legitimate move, repeat the same until the board is solved.
              If not select another number from the cell.
              Assigning a number will create one recursion and create a new Sudoku board 
              because it is easier to backtrack.
 * Argument depth is for logging purposes
 */

Sudoku* solve(Sudoku* S, int depth) {
   // current branch depth and number of unassigned cells
   // cout << "depth : " << depth << ", unassigned : " << S->unassigned() <<  endl;
   // base case
   if (S == nullptr || S->is_solved()) 
      return S;
   
   // cell with least possible #
   int least_k[BOARD_SIZE];
   S->least_count(least_k);

   int k = least_k[0];      
   // possible number of that cell
   Possible p = S->possible(k);
   for (int i = 1; i <= BOARD_WIDTH; ++i) {
      // if number i is possible on _cell[k]
      if (p.is_on(i)) {
         attempt_per_depth[depth]++;
         // cout << "Depth: " << depth << ", Attempts: " << attempt_per_depth[branches] << endl;
         // make new Sudoku board
         Sudoku* S1 = new Sudoku(*S);
         // if there is no conflict in assign number i to cell[k] go deeper
         if (S1->assign(k, i)) 
            if (auto S2 = solve(S1, depth + 1)) 
               return S2;
         delete S1;
      }
   }
   //will return null string whene there is no solution
   return {};
}

// Random Sudoku generator shuffles the Sudoku cells randomly and assign random number to cells
// N_ASSIGN : number of cells to be assigne
// DIFF : number of assigned unique numbers

/*
 * void rand_generator(int num) : Generates random Sudoku boards
   Algorithm: (1) Randomly select cells and assign a number randomly with at least DIFF different numbers
              and N_ASSIGN number of assigned cell.
              If the assignment is invalid do it until it's valid.
              (2) After generating a partially field board, solve it. 
              (3) After solving a board, randomly remove cells at each 10% it will be saved to a file.
 */

void rand_generator(int num) {
   int val;
   int cell_num[BOARD_SIZE];
   string cell_val[BOARD_SIZE];
   bool err = false;
   ofstream myfile[10];
   string ret;

   for (int i = 0; i < 9; ++i) 
      myfile[i].open(to_string(N) + "Sudoku" + to_string(100-(i+1)*10) + "%.txt", fstream::app);

   for (int total = 0 ; total < num; ++total) {

      for (int i = 0; i < BOARD_SIZE; ++i) 
         cell_num[i] = i;
      
      // count unassigned numbers, at least 8 different numbers should be assigned to 9x9 Sudoku
      Possible* unassigned_nums = new Possible();
      int assigned_num[N_ASSIGN];

      // shuffle cell numbers to assign values
      shuffle(cell_num, cell_num + BOARD_SIZE, default_random_engine(time(0)));

      Sudoku* s;
      // should have at least 8 different number for 9x9 Sudoku.(9 - unassigned = 8)
      while( BOARD_WIDTH - unassigned_nums->count() < DIFF) {
         s = new Sudoku();
         // assign at least N_ASSIGN(=17) numbers for 9x9 Sudoku
         for (int i = 0; i < N_ASSIGN; ++i) {
            val = rand() % (BOARD_WIDTH) + 1;
            // cout << val << endl;
            // elimninate assigned value
            unassigned_nums->eliminate(val);
            assigned_num[i] = val;
            // check if there is any conflict
            if (!s->assign(cell_num[i], val)) {
               // cerr << "error" << endl;
               err = true;
               continue;
            }
         }
         // if error occured make new object and start again
         if (err) {
            delete unassigned_nums;
            unassigned_nums = new Possible();
            err = false;
            delete s;
         }
      }
      delete unassigned_nums;

      Sudoku* solved = solve(s, 0);

      shuffle(cell_num, cell_num + BOARD_SIZE, default_random_engine(time(0)));

      for (int i = 0; i <= BOARD_SIZE * 0.9 ; ++i) {
         solved->_cells[cell_num[i]] = Possible(SET_BIT);
         if (i % (BOARD_SIZE/10) == 1) {
            myfile[i / (BOARD_SIZE/10)] << solved->input_string();
         }
      }
      cout << total << endl;
   }

   for (int i = 0; i < 10; ++i) 
      myfile[i].close();
}

int main(int argc, char *argv[]) {
   Sudoku::init();
   string line;
   int it;
   double total_time;
   double time_taken;
   clock_t start, end;

   srand(time(NULL));

   // rand_generator(100);
   getline(cin, line);

   start = clock();
   auto S = solve(new Sudoku(line), 1);
   end = clock();    
   time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
   S->write(cout);
   printf("Time Taken: %f\n", time_taken);

   // for (int i = 0; i < BOARD_SIZE; ++i)
   //    cout << i << ": " << attempt_per_depth[i] << endl;

   return 0;
}
