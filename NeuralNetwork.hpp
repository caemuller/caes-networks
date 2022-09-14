#include <iostream>
#include <vector>
#include <math.h>
#include "node.hpp"
#include "layer.hpp"
#include "Matrix.hpp"

class NeuralNetwork{
    public:
        NeuralNetwork(std::vector<int> topology){
            this->topology = topology;
            this->topology_size = topology.size();
            for(int i=0;i<this->topology_size;i++){
                Layer *l = new Layer(topology.at(i));
                this->layers.push_back(l);
            }
    
            for(int i=0;i<this->topology_size-1;i++){
                Matrix *m = new Matrix(topology.at(i), topology.at(i+1), true);
                this->weights.push_back(m);
            }
        }

        void set_input(std::vector<double> input){
            this->input = input;
            for(int i=0;i<input.size();i++)
                this->layers.at(i)->set_value(i, input.at(i));
        }

        void set_Node_val(int idx_layer, int idx_node, double val){
            this->layers.at(idx_layer)->set_value(idx_node, val);
        }
        Matrix* getNodeMatrix(int idx){
            return this->layers.at(idx)->vals_to_matrix();
        }
        Matrix* getactNodeMatrix(int idx){
            return this->layers.at(idx)->actvals_to_matrix();
        }
        Matrix* getdxNodeMatrix(int idx){
            return this->layers.at(idx)->dxvals_to_matrix();
        }
        Matrix* getweighNodeMatrix(int idx){
            return this->weights.at(idx);
        }
        Matrix* multiplyMatrix(Matrix *a, Matrix *b){
            Matrix* c = new Matrix(a->get_rows(), b->get_cols(), false);
            #pragma omp parallel for
            for(int i=0;i<a->get_rows();i++)
                for(int j=0;j<b->get_cols();j++)
                    for(int k=0;k<b->get_rows();k++){
                        double p = a->get_val(i, k) * b->get_val(k, j);
                        double val = c->get_val(i, j) + p;
                        c->set_value(i, j, val);
                    }
        }

        void show_nn(){
            for(int i=0;i<this->layers.size();i++){
                std::cout << "layer " << i << std::endl;
                if(!i){
                    Matrix *m = this->layers.at(i)->vals_to_matrix();
                    m->show_matrix();
                } else{
                    Matrix *m = this->layers.at(i)->actvals_to_matrix();
                    m->show_matrix();
                }
                std::cout << "=======================" << std::endl;
                if(i < this->layers.size()-1){
                    std::cout << "weights " << i << std::endl;
                    this->getweighNodeMatrix(i)->show_matrix();
                }
                std::cout << "=======================" << std::endl;
                std::cout << "-----------------------" << std::endl;
            }
        }


        void feedForward(){
            for(int i=0;i<this->layers.size()-1;i++){
                Matrix *a = this->getNodeMatrix(i);
                if(i)
                    a = this->getactNodeMatrix(i);
                Matrix *b = this->getweighNodeMatrix(i);
                Matrix *c = multiplyMatrix(a, b);
                for(int ci=0;ci<c->get_cols();ci++)
                    this->set_Node_val(i+1, ci, c->get_val(0, ci));


            }
        }

    private:
        int topology_size;
        std::vector<int> topology;
        std::vector<Layer*> layers;
        std::vector<Matrix*> weights;
        std::vector<double> input;
};
