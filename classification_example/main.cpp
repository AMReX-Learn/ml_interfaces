#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include <torch/torch.h>

using namespace amrex;

// MultiFab Dataset for NN
class MFDataset : public torch::data::Dataset<MFDataset>
{
    private:
        std::vector<std::tuple< torch::Tensor, int64_t >> mfv_;

    public:
        explicit MFDataset(Vector<MultiFab>& mfv, int max_grid_size, int n_cell)
        // Load mf data and label
        {
            for (int i=0; i<mfv.size(); i++){
                for (MFIter mfi(mfv[i]); mfi.isValid(); ++mfi) // Loop over grids
                {
                    const Box& box = mfi.validbox();
                    const auto lo = lbound(box);
                    const auto hi = ubound(box);
                    Array4<Real      > const& f = mfv[i].array(mfi);
                    float auxPtr[mfv[i].nComp()*max_grid_size*max_grid_size];
                    for (int nc=0; nc<mfv[i].nComp(); nc++){
                      int k = 0; 
                      for   (int j = lo.y; j <= hi.y; ++j) {
                        for (int i = lo.x; i <= hi.x; ++i) {
                          float f_aux = (float) f(i,j,0,nc);
                          auxPtr[k+(nc*max_grid_size*max_grid_size)] = f_aux;
                          k++;
                        }
                      }
                    }
                    torch::Tensor tensor = torch::from_blob(auxPtr, {max_grid_size,max_grid_size,3}).clone();
                    int64_t aux=0;
                    if (i%2==1) aux=1;
                    mfv_.push_back(std::make_tuple(tensor,aux));
                }
            }
        };

        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {

            torch::Tensor mf_tensor = std::get<0>(mfv_[index]);
            int64_t label = std::get<1>(mfv_[index]);
            mf_tensor = mf_tensor.permute({2, 0, 1}); // convert to CxHxW

            torch::Tensor label_tensor = torch::full({1}, label);
            return {mf_tensor, label_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {

            return mfv_.size();
        };
};

struct ConvNetImpl : public torch::nn::Module 
{
    ConvNetImpl(int64_t channels, int64_t height, int64_t width) 
        : conv1(torch::nn::Conv2dOptions(3 /*input channels*/, 8 /*output channels*/, 5 /*kernel size*/).stride(2)),
          conv2(torch::nn::Conv2dOptions(8, 16, 3).stride(2)),
          
          n(GetConvOutput(channels, height, width)),
          lin1(n, 32),
          lin2(32, 2 /*number of output classes (circles and squares)*/) {

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("lin1", lin1);
        register_module("lin2", lin2);
    };

    // Implement the forward method.
    torch::Tensor forward(torch::Tensor x) {

        x = torch::relu(torch::max_pool2d(conv1(x), 2));
        x = torch::relu(torch::max_pool2d(conv2(x), 2));

        // Flatten.
        x = x.view({-1, n});

        x = torch::relu(lin1(x));
        x = torch::log_softmax(lin2(x), 1/*dim*/);

        return x;
    };

    // Get number of elements of output.
    int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {

        torch::Tensor x = torch::zeros({1, channels, height, width});
        x = torch::max_pool2d(conv1(x), 2);
        x = torch::max_pool2d(conv2(x), 2);

        return x.numel();
    }

    torch::nn::Conv2d conv1, conv2;
    int64_t n;
    torch::nn::Linear lin1, lin2;
};

TORCH_MODULE(ConvNet);

#if 0
// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
#endif

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        int n_cell = 128;
        int max_grid_size = 32;
        int which_geom = 0;
        int size_train = 512;

        // read parameters
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
            pp.query("which_geom_predict", which_geom);
            pp.query("size_train", size_train);
        }

        Geometry geom;
        {
            RealBox rb({AMREX_D_DECL(-2.0,-2.0,-2.0)}, {AMREX_D_DECL(2.0,2.0,2.0)}); // physical domain
            Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(false, false, false)};
            Geometry::Setup(&rb, 0, is_periodic.data());
            Box domain(IntVect(0), IntVect(n_cell-1));
            geom.define(domain);            
        }

        /////////// GENERATING MULTIFAB DATASET ///////////////////////////////////////
  
        Vector<MultiFab> mfv(size_train);
        ResetRandomSeed(time(0));
        for (int i=0; i<size_train; i++){
            if (i%2 == 0) {
                //CIRCLES!!!!
                EB2::SphereIF sphere(0.5, {AMREX_D_DECL(-1.5+(3.0*amrex::Random()),-1.5+(3.0*amrex::Random()),-1.5+(3.0*amrex::Random()))}, false);
                //EB2::SphereIF sphere(0.5, {AMREX_D_DECL(1.0,1.0,1.0)}, false);
                auto gshop = EB2::makeShop(sphere);
                EB2::Build(gshop, geom, 0, 0);
            } else {
                //SQUARES!!!!!
                Real rand = amrex::Random();
                EB2::BoxIF cube({AMREX_D_DECL(-1.0+rand,-1.0+rand,-1.0+rand)}, {AMREX_D_DECL(rand,rand,rand)}, false);
                //EB2::BoxIF cube({AMREX_D_DECL(-1.0,-1.0,-1.0)}, {AMREX_D_DECL(0.0,0.0,0.0)}, false);
                auto gshop = EB2::makeShop(cube);
                EB2::Build(gshop, geom, 0, 0);
            }
            BoxArray ba(geom.Domain());
            ba.maxSize(max_grid_size);
            DistributionMapping dm{ba};
 
            std::unique_ptr<EBFArrayBoxFactory> factory
                = amrex::makeEBFabFactory(geom, ba, dm, {2,2,2}, EBSupport::full);
 
            mfv[i].define(ba, dm, 3, 0, MFInfo(), *factory);
            mfv[i].setVal(0.0);
            EB_set_covered(mfv[i], 0, mfv[i].nComp(), mfv[i].nGrow(), 255.0);
        }
        //EB_WriteSingleLevelPlotfile("plt_train", mfv[0], {"rho"}, geom, 0.0, 0);

        //std::cout<<"Tensor example from PYTORCH!"<<std::endl;
        //torch::Tensor tensor = torch::rand({2, 3});
        //std::cout << tensor << std::endl;

        // Passing MFs info to Dataset
        auto data_set = MFDataset(mfv,max_grid_size,n_cell).map(torch::data::transforms::Stack<>());
        // Generate a data loader.
        int64_t batch_size = 32;
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            data_set,
            batch_size);

        /////////// TRAINING MULTIFAB DATASET ///////////////////////////////////////

        // Load the model.
        ConvNet model(3/*channel*/, max_grid_size/*height*/, max_grid_size/*width*/);
    
        // Chose and optimizer.
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    
        // Train the network.
        int64_t n_epochs = 10;
        int64_t log_interval = 10;
        int dataset_size = data_set.size().value();
    
        // Record best loss.
        float best_mse = std::numeric_limits<float>::max();
    
        for (int epoch = 1; epoch <= n_epochs; epoch++) {
    
            // Track loss.
            size_t batch_idx = 0;
            float mse = 0.; // mean squared error
            int count = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (auto& batch : *data_loader) {
                auto imgs = batch.data;
                auto labels = batch.target.squeeze();
    
                imgs = imgs.to(torch::kF32);
                labels = labels.to(torch::kInt64);
                // Reset gradients.
                optimizer.zero_grad();
                // Execute the model on the input data.
                auto output = model(imgs);
                // Compute a loss value to judge the prediction of our model.
                auto loss = torch::nll_loss(output, labels);
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
    
                mse += loss.template item<float>();
    
                batch_idx++;
                if (batch_idx % log_interval == 0) 
                {
                    std::printf(
                    "\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f",
                    epoch,
                    n_epochs,
                    batch_idx * batch.data.size(0), 
                    dataset_size,
                    loss.template item<float>());
                }
    
                count++;
            }
    
            mse /= (float)count;
            printf(" Mean squared error: %f\n", mse);   
    
            if (mse < best_mse)
            {
                //torch::save(model, "../best_model.pt");
                best_mse = mse;
            }
        } 
    
        /////////// MULTIFAB PREDICTION (CLASSIFICATION) ///////////////////////////////

        MultiFab mfp;
        {
            ResetRandomSeed(time(0));
            if (which_geom == 0) {
                std::cout<<"Creating a circle with AMReX!"<<std::endl;
                Real randx = amrex::Random();
                Real randy = amrex::Random();
                Real randz = amrex::Random();
                EB2::SphereIF sphere(0.5, {AMREX_D_DECL(-1.5+(3.0*randx),-1.5+(3.0*randy),-1.5+(3.0*randz))}, false);
                //EB2::SphereIF sphere(0.5, {AMREX_D_DECL(1.0,1.0,1.0)}, false);
                auto gshop = EB2::makeShop(sphere);
                EB2::Build(gshop, geom, 0, 0);
            } else {
                std::cout<<"Creating a square with AMReX! "<<std::endl;
                Real rand = amrex::Random();
                EB2::BoxIF cube({AMREX_D_DECL(-1.0+rand,-1.0+rand,-1.0+rand)}, {AMREX_D_DECL(rand,rand,rand)}, false);
                //EB2::BoxIF cube({AMREX_D_DECL(-1.0,-1.0,-1.0)}, {AMREX_D_DECL(0.0,0.0,0.0)}, false);
                auto gshop = EB2::makeShop(cube);
                /*Real rotation  = (45./180.)*M_PI;
                int rotation_axe  = 2;
                auto cube_rotated = EB2::rotate(cube, rotation, rotation_axe);
                auto gshop = EB2::makeShop(cube_rotated);*/
                EB2::Build(gshop, geom, 0, 0);
            }
            BoxArray ba(geom.Domain());
            ba.maxSize(max_grid_size);
            DistributionMapping dm{ba};

            std::unique_ptr<EBFArrayBoxFactory> factory
                = amrex::makeEBFabFactory(geom, ba, dm, {2,2,2}, EBSupport::full);

            mfp.define(ba, dm, 3, 0, MFInfo(), *factory);
            mfp.setVal(0.0);
            EB_set_covered(mfp, 0, mfp.nComp(), mfp.nGrow(), 255.0);
        }

        VisMF::Write(mfp,"plt_pred");

        torch::Tensor tensorp;
        for (MFIter mfi(mfp); mfi.isValid(); ++mfi) // Loop over grids
        {   
            const Box& box = mfi.validbox();
            const auto lo = lbound(box);
            const auto hi = ubound(box);
            Array4<Real      > const& f = mfp.array(mfi);
            float auxPtr[mfp.nComp()*max_grid_size*max_grid_size];
            for (int nc=0; nc<mfp.nComp(); nc++){
              int k = 0;
              for   (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                  float f_aux = (float) f(i,j,0,nc);
                  auxPtr[k+(nc*max_grid_size*max_grid_size)] = f_aux;
                  k++;
                }
              }
            }
            tensorp = torch::from_blob(auxPtr, {1,max_grid_size,max_grid_size,3}).clone();
        }

        tensorp = tensorp.permute({0, 3, 1, 2}); // convert to CxHxW
        tensorp = tensorp.to(torch::kF32);
    
        // Predict the probabilities for the classes.
        torch::Tensor log_prob = model(tensorp);
        torch::Tensor prob = torch::exp(log_prob);
    
        printf("Probability of being\n\
        a circle = %.2f percent\n\
        a square = %.2f percent\n", prob[0][0].item<float>()*100., prob[0][1].item<float>()*100.); 

#if 0
        std::cout<<"NN example from PYTORCH!"<<std::endl;

	// Create a new Net.
	auto net = std::make_shared<Net>();

	// Create a multi-threaded data loader for the MNIST dataset.
	auto data_loader = torch::data::make_data_loader(
			torch::data::datasets::MNIST("../data").map(
				torch::data::transforms::Stack<>()),
			/*batch_size=*/64);

	// Instantiate an SGD optimization algorithm to update our Net's parameters.
	torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

	for (size_t epoch = 1; epoch <= 10; ++epoch) {
	   size_t batch_index = 0;
	   // Iterate the data loader to yield batches from the dataset.
	   for (auto& batch : *data_loader) {
	      // Reset gradients.
	      optimizer.zero_grad();
	      // Execute the model on the input data.
	      torch::Tensor prediction = net->forward(batch.data);
	      // Compute a loss value to judge the prediction of our model.
	      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
	      // Compute gradients of the loss w.r.t. the parameters of our model.
	      loss.backward();
	      // Update the parameters based on the calculated gradients.
	      optimizer.step();
	      // Output the loss and checkpoint every 100 batches.
	      if (++batch_index % 100 == 0) {
	         std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
	                   << " | Loss: " << loss.item<double>() << std::endl;
	         // Serialize your model periodically as a checkpoint.
	         torch::save(net, "net.pt");
	      }
	   }
	}
#endif

    }

    amrex::Finalize();
}
