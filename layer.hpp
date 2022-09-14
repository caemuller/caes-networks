#include <iostream>
#include <vector>
#include <math.h>
#include "node.hpp"
#include "Matrix.hpp"

class Layer{
    public:
        Layer(int size){
            this->size = size;
            for(int i=0;i<size;i++)
            {
                Node *n = new Node(0.0);
                this->nodes.push_back(n);
            }
        }

        void set_value(int i, double val){
            this->nodes.at(i)->set_val(val);
        }

        Matrix *vals_to_matrix(){
            Matrix *m = new Matrix(1, this->nodes.size(), false);
            for(int i=0;i<this->nodes.size();i++)
                m->set_value(0, i, this->nodes.at(i)->get_val());
            return m;
        }

        Matrix *actvals_to_matrix(){
            Matrix *m = new Matrix(1, this->nodes.size(), false);
            for(int i=0;i<this->nodes.size();i++)
                m->set_value(0, i, this->nodes.at(i)->get_actval());
            return m;    
        }

        Matrix *dxvals_to_matrix(){
            Matrix *m = new Matrix(1, this->nodes.size(), false);
            for(int i=0;i<this->nodes.size();i++)
                m->set_value(0, i, this->nodes.at(i)->get_dxval());
            return m;
        }
    private:
        int size;
        std::vector<Node*> nodes;
};