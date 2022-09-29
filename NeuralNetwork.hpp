#include <iostream>
#include <vector>
#include <math.h>
#include "Layer.hpp"

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

        // getters

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
        double get_total_error(){return this->error;}
        std::vector<double> get_errors(){return this->errors;}

        //setter

        void set_input(std::vector<double> input){
            this->input = input;
            for(int i=0;i<input.size();i++)
                this->layers.at(i)->set_value(i, input.at(i));
        }

        void set_Node_val(int idx_layer, int idx_node, double val){
            this->layers.at(idx_layer)->set_value(idx_node, val);
        }

        void set_errors(){
            this->error = 0;
            int out_layer = this->layers.size()-1;
            std::vector<Node*> output_nodes = this->layers.at(out_layer)->get_nodes();
            for(int i=0;i<this->target.size();i++){
                double aux_err = (output_nodes.at(i)->get_actval() - target.at(i));
                errors.at(i) = aux_err;
                this->error += aux_err;
            }
            hist_errors.push_back(this->error);
        }

        // methods
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


        void feed_forward(){
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

        void back_propagation(){
            std::vector<Matrix*> new_weights;
            Matrix* gradients;
            int output_idx = this->layers.size()-1;
            Matrix *dx_y_to_z = this->layers.at(output_idx)->dxvals_to_matrix();
            Matrix* gradients_yz = new Matrix(1, this->layers.at(output_idx)->get_nodes().size(), false);
            Matrix* left_nodes;

            for (int i=0;i<this->errors.size();i++)
            {
                double dx = dx_y_to_z->get_val(0, i);
                double er = this->errors.at(i);
                double gr = dx * er;
                gradients_yz->set_value(0, i, gr);   
            }

            int last_hidden = output_idx-1;
            Layer* last_hidden_layer = this->layers.at(last_hidden);
            Matrix* weights_out_hidden = this->getweighNodeMatrix(last_hidden);
            Matrix* delta_out = multiplyMatrix(gradients_yz->transpose(), last_hidden_layer->actvals_to_matrix())->transpose();
            Matrix* new_weights_out_hidden = new Matrix(delta_out->get_rows(), delta_out->get_cols(), false);

            for(int i=0;i<delta_out->get_rows();i++)
                for(int j=0;j<delta_out->get_cols();j++)
                    new_weights_out_hidden->set_value(i, j, weights_out_hidden->get_val(i, j) - delta_out->get_val(i, j));
 
            new_weights.push_back(new_weights_out_hidden);
            gradients = new Matrix(gradients_yz->get_rows(), gradients_yz->get_cols(), false);

            for(int i=0;i<gradients_yz->get_rows();i++)
                for(int j=0;j<gradients_yz->get_cols();j++)
                    gradients->set_value(i, j, gradients_yz->get_val(i, j));

            for(int i=last_hidden;i>0;i--)
            {
                Layer* l = this->layers.at(i);
                Matrix* derived = l->dxvals_to_matrix();
                Matrix* derived_gradients = new Matrix(1, l->get_nodes().size(), false);
                Matrix* activated_hidden = l->actvals_to_matrix();

                Matrix* weight_matrix = this->weights.at(i);

                for(int j=0;j<weight_matrix->get_rows();j++){
                    double sum=0;
                    for(int k=0;k<weight_matrix->get_cols();k++){
                        sum += gradients->get_val(j, k) * weight_matrix->get_val(j, k);
                    }
                    derived_gradients->set_value(0, j, sum*activated_hidden->get_val(0, j));
                }

                if(!i-1)
                    left_nodes = this->layers.at(i-1)->actvals_to_matrix();
                else
                    left_nodes = this->layers.at(0)->vals_to_matrix();

                Matrix* delta_weights = multiplyMatrix(derived_gradients->transpose(), left_nodes)->transpose();
                Matrix* new_weights_hidden = new Matrix(delta_weights->get_rows(), delta_weights->get_cols(), false);
                for(int j=0;j<new_weights_hidden->get_rows();j++)
                    for(int k=0;k<new_weights_hidden->get_cols();k++)
                        new_weights_hidden->set_value(j, k, this->weights.at(i-1)->get_val(j, k) - delta_weights->get_val(j, k));

                new_weights.push_back(new_weights_hidden);
                gradients = new Matrix(derived_gradients->get_rows(), derived_gradients->get_cols(), false);
                for(int j=0;j<gradients->get_rows();j++)
                    for(int k=0;k<gradients->get_cols();k++)
                        gradients->set_value(j, k, derived_gradients->get_val(j, k));
            }
        }

    private:
        int topology_size;
        std::vector<int> topology;
        std::vector<Layer*> layers;
        std::vector<Matrix*> weights;
        std::vector<Matrix*> gradients;
        std::vector<double> input;
        std:: vector<double> target;
        std:: vector<double> errors;
        std:: vector<double> hist_errors;
   
        double error;
};
