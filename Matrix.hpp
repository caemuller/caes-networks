#include <iostream>
#include <vector>
#include <math.h>
#include "node.hpp"
#include <cstdlib>
#include <random>
#include <iomanip>

using std::setprecision;

class Matrix{
    public:
        Matrix(int rows, int cols, bool random){
            for(int i=0;i<rows;i++)
            {
                std::vector<double> col_values;
                for(int j=0;j<cols;j++)
                {
                    double r=0.0;
                    if(random)
                        r = this->rng();
                    col_values.push_back(r);
                }
                this->vals.push_back(col_values);
            }
        }

        void show_matrix(){
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++)
                    std::cout << this->vals.at(i).at(j) << "\t";
                std::cout << std::endl;
            }
        }

        double rng(){
            std::random_device rd;
            std::default_random_engine eng(rd());
            std::uniform_real_distribution<float> distr(0, 1);
        }

        Matrix *transpose(){
            Matrix *m = new Matrix(this->cols, this->rows, false);
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    m->set_value(j, i, this->get_val(i, j));
                }
            }
            return m;
        }

        //getter
        int get_rows(){return this->rows;}
        int get_cols(){return this->cols;}
        double get_val(int row, int col){return this->vals.at(row).at(col);}
        //setter
        void set_value(int r, int c, double num){
            this->vals.at(r).at(c) = num;
        }


    private:
        int rows;
        int cols;
        std::vector<std::vector<double>> vals;
};