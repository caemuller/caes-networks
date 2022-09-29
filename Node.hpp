#include <iostream>
#include <math.h>

class Node{
    public:
        Node(double val){
            this->val = val;
            activate();
            dx();
        }

        void activate(){
            // I'm using the fast sigmoid function here :)
            this->actval = this->val / (1 + abs(this->val));
        }
        
        void dx(){
            //deriving the fast sigmoid func above
            this->dxval = this->actval * (1 - this->actval);
        }
        
        //getterfunc
        double get_val(){return this->val;}
        double get_actval(){return this->actval;}
        double get_dxval(){return this->dxval;}

        // setter function
        void set_val(double val){
            this->val = val;
            activate();
            dx();
        }

    private:
        double val;
        double actval;
        double dxval;
};