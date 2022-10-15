#include "array"
#include "algorithm"
#include "numeric"
#include "cmath"
#include "gurobi_c++.h"
#include "rapidcsv.h"
#include "filesystem"

template<int32_t num_clients>
class Model
{
private:
     rapidcsv::Document data_frame;
     int32_t num_cars;
     int32_t capacity;

public:
     Model(rapidcsv::Document _data_frame, int32_t _num_cars, int32_t _capacity) : 
     data_frame(_data_frame), num_cars(_num_cars), capacity(_capacity){};

private:
     template<int32_t _num_clients>
     std::array<std::array<int32_t, _num_clients+1>, _num_clients+1> get_distance()
     {
          auto coord_x = data_frame.GetColumn<int32_t>("x");
          auto coord_y = data_frame.GetColumn<int32_t>("y");
          std::array<std::array<int32_t, _num_clients+1>, _num_clients+1> distance;
          for (size_t i=0; i<num_clients+1; i++)
               for (size_t j=0;j<num_clients+1;j++)
               {
                    distance[i][j] = std::lround(std::pow(std::pow(coord_x[i]-coord_x[j],2) + std::pow(coord_y[i]-coord_y[j],2),0.5));                                         
               }
          return distance;
     } 

public:
     void solve_task()
     {
          try
          {
               GRBEnv env = GRBEnv(true);
               env.set(GRB_DoubleParam_TimeLimit, 60);
               env.start();
               GRBModel model = GRBModel(env);

               auto income = data_frame.GetColumn<int32_t>("income");
               income.erase(income.begin(), income.begin()+1);
               auto demand = data_frame.GetColumn<int32_t>("demand");
               demand.erase(demand.begin(), demand.begin()+1);

               auto distance = get_distance<num_clients>();
               GRBLinExpr distance_cost = 0;
               GRBLinExpr total_profit = 0;

               GRBVar is_delivered[num_clients];
               for (auto &i : is_delivered)
                    i = model.addVar(0,1,1,GRB_BINARY);

               //кол-во груза i авто j клиенту
               GRBVar cars_to_clients[num_cars][num_clients];
               for (auto &i : cars_to_clients)
                    for (auto &j : i)
                         j = model.addVar(0, capacity , 1, GRB_INTEGER);
               //путь i машины через j и k вершины
               GRBVar car_path[num_cars][num_clients + 1][num_clients + 1];
               for (auto &i : car_path)
                    for (auto &j : i)
                         for (auto &k : j)
                              k = model.addVar(0, 1, 1, GRB_BINARY);

               GRBVar taked_vertices[num_cars][num_clients + 1];
               for (auto &i : taked_vertices)
                    for (auto &j : i)
                         j = model.addVar(0, 1, 1, GRB_BINARY);

               for (size_t i=0; i<num_clients; i++)
               {
                    GRBLinExpr* sum = new GRBLinExpr(0.0f);
                    for (size_t j=0; j<num_cars; j++)
                         *sum += cars_to_clients[j][i];
                    model.addConstr(is_delivered[i] >= *sum/demand[i]);
               }
               
               //Антимонополия
               GRBLinExpr* sum = new GRBLinExpr(0.0f);
               for (size_t i=0; i<num_clients; i++)
                    *sum += is_delivered[i];

               model.addConstr(*sum <= 0.5*num_clients);

               for (size_t i=0; i<num_cars; i++)
               {               
                    GRBVar edges[num_clients + 1][num_clients + 1];

                    for (auto &i : edges)
                         for (auto &j : i)
                              j = model.addVar(0, 1000, 1,  GRB_INTEGER);

                    model.addConstr(taked_vertices[i][0] == 1);

                    for (size_t j=0; j<num_clients; j++)
                    {
                         model.addConstr(taked_vertices[i][j+1]*demand[j] >= cars_to_clients[i][j]);
                    }
                    for (size_t j=0; j<num_clients+1; j++)
                    {
                         GRBLinExpr* ingoing = new GRBLinExpr(0.0f);
                         for (size_t l =0; l<num_clients+1; l++)
                              *ingoing+= car_path[i][l][j];
                         model.addConstr(*ingoing == taked_vertices[i][j]);

                         GRBLinExpr* outcoming = new GRBLinExpr(0.0f);
                         for (size_t l =0; l<num_clients+1; l++)
                              *outcoming+= car_path[i][j][l];
                         model.addConstr(*outcoming == taked_vertices[i][j]);

                         model.addConstr(car_path[i][j][j] == 0);
                    }
                         
                    for (size_t j=0; j<num_clients+1; j++)
                    {
                         for (size_t k=0; k<num_clients+1; k++)
                         {
                              model.addConstr(edges[j][k] - num_clients*car_path[i][j][k] <= 0);     
                         }
                    }

                    GRBLinExpr* temp1 = new GRBLinExpr(0.0f);
                    for (size_t j=0; j< num_clients+1; j++)
                         *temp1+= edges[0][j];
                    GRBLinExpr* temp2 = new GRBLinExpr(0.0f);
                    for (size_t j=0; j<num_clients+1; j++)
                         *temp2+= taked_vertices[i][j];

                    model.addConstr(*temp1 - *temp2 == -1);

                    for (size_t j=1; j<num_clients+1;j++)
                    {
                         GRBLinExpr* sum1 = new GRBLinExpr(0.0f);
                         GRBLinExpr* sum2 = new GRBLinExpr(0.0f);

                         for (size_t k=0; k<num_clients+1; k++)
                         {
                              *sum1+=edges[j][k];
                              *sum2+=edges[k][j];
                         }
                         model.addConstr(*sum1 - *sum2 == -taked_vertices[i][j]);
                    }

                    //Вычисляем стоимость бензина i машине
                    for (size_t j=0; j<num_clients+1; j++)
                    {
                         for (size_t k=0; k<num_clients+1; k++)
                              distance_cost+=car_path[i][j][k]*distance[j][k];
                    }

                    //Сколько заработали на i машине
                    for (size_t j=0; j<num_clients;j++)
                         total_profit+=(cars_to_clients[i][j]/demand[j])*income[j];

                    //Смотрим нет ли перевеса на i машине
                    GRBLinExpr* weight =  new GRBLinExpr(0.0f);
                    for (size_t j =0; j<num_clients; j++)
                         *weight+=cars_to_clients[i][j];

                    model.addConstr(*weight <= capacity);
               }

               model.setObjective(total_profit - distance_cost, GRB_MAXIMIZE);
               model.optimize();

               std::cout << "distance cost = " << distance_cost.getValue() << "\n";
               std::cout << "total profit = " << total_profit.getValue() << "\n";
          }
          catch (GRBException e)
          {
               std::cout << "Error code = " << e.getErrorCode() << std::endl;
               std::cout << e.getMessage() << std::endl;
          }
          catch (...)
          {
               std::cout << "Exception during optimization" << std::endl;
          }
     }
};

int main(int argc, char *argv[])
{
     namespace fs = std::filesystem;
     rapidcsv::Document doc1(fs::current_path().string() + "/dataset/n15-k4-Q20.csv");
     rapidcsv::Document doc2(fs::current_path().string() + "/dataset/n39-k6-Q40.csv");
     auto model1 = Model<15>(doc1, 4, 20);
     auto model2 = Model<39>(doc2, 6, 40);
     model2.solve_task();
}
