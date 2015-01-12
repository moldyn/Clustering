
#include "network_builder.hpp"

#include "tools.hpp"
#include "logger.hpp"
#include "embedded_cytoscape.hpp"

#include <fstream>
#include <set>
#include <unordered_set>

#include <boost/program_options.hpp>
#include <omp.h>


namespace {

  constexpr bool fuzzy_equal(float a, float b, float prec) {
    return (a <= b + prec) && (a >= b - prec);
  }

  // overload output operator for Node-serialization
  // (producing string representation of node + edges to children)
  std::ostream& operator<<(std::ostream& os, const Node& n) {
    float log_pop;
    if (n.pop <= 0) {
      log_pop = log(1);
    } else {
      log_pop = log(n.pop);
    }
    // print node itself
    os << Clustering::Tools::stringprintf("{group:'nodes',id:'n%d',position:{x:%d,y:%d},data:{id:'n%d',pop:%d,fe:%f,info:'%d: fe=%0.2f, pop=%d',logpop:%0.2f}},\n",
                                          n.id, n.pos_x, n.pos_y, n.id, n.pop, n.fe, n.id, n.fe, n.pop, log_pop);
    // print edges from node's children to node
    for (auto& id_child: n.children) {
      std::size_t cid = id_child.first;
      os << Clustering::Tools::stringprintf("{group:'edges',data:{id:'e%d_%d',source:'n%d',target:'n%d'}},\n", cid, n.id, cid, n.id);
    }
    return os;
  }
  
  Node::Node() {}
  
  Node::Node(std::size_t _id,
             float _fe,
             std::size_t _pop)
    : id(_id)
    , fe(_fe)
    , pop(_pop) {
  }
  
  Node* Node::find_parent_of(std::size_t search_id) {
    if (this->children.count(search_id)) {
      return this;
    } else {
      for (auto& id_child: children) {
        Node* cc = id_child.second.find_parent_of(search_id);
        if (cc) {
          return cc;
        }
      }
    }
    return NULL;
  }
  
  void Node::set_pos(int x, int y) {
    this->pos_x = x;
    this->pos_y = y;
    std::vector<int> width_children;
    int total_width = 0;
    for (auto& id_child: children) {
      width_children.push_back(id_child.second.subtree_width());
      total_width += id_child.second.subtree_width();
    }
    // set pos for every child recursively,
    // regarding proper segmentation of horizontal space.
    int cur_x = (x-0.5*total_width);
    for (auto& id_child: children) {
      int stw = id_child.second.subtree_width();
      id_child.second.set_pos(cur_x + 0.5*stw, y + VERTICAL_SPACING);
      cur_x += stw;
    }
  }

  int Node::subtree_width() {
    // check for backtracked value
    // and compute if not existing.
    if ( ! this->_subtree_width) {
      int self_width = 10 + 2*HORIZONTAL_SPACING; //TODO get from size
      if (children.empty()) {
        this->_subtree_width = self_width;
      } else {
        int sum = 0;
        for (auto& id_child: children) {
          sum += id_child.second.subtree_width();
        }
        if (sum > self_width) {
          this->_subtree_width = sum;
        } else {
          this->_subtree_width = self_width;
        }
      }
    }
    return this->_subtree_width;
  }

  void Node::print_subtree(std::ostream& os) {
    for (auto& id_child: children) {
      id_child.second.print_node_and_subtree(os);
    }
  }

  void Node::print_node_and_subtree(std::ostream& os) {
    os << (*this) << std::endl;
    this->print_subtree(os);
  }


  void
  save_network_links(std::string fname,
                     std::map<std::size_t, std::size_t> network) {
    Clustering::logger(std::cout) << "saving links" << std::endl;
    std::ofstream ofs(fname);
    if (ofs.fail()) {
      std::cerr << "error: cannot open file '" << fname << "' for writing." << std::endl;
      exit(EXIT_FAILURE);
    } else {
      for (auto p: network) {
        ofs << p.second << " " << p.first << "\n";
      }
    }
  }
  
  void
  save_node_info(std::string fname,
                 std::map<std::size_t, float> free_energies,
                 std::map<std::size_t, std::size_t> pops) {
    Clustering::logger(std::cout) << "saving nodes" << std::endl;
    std::ofstream ofs(fname);
    if (ofs.fail()) {
      std::cerr << "error: cannot open file '" << fname << "' for writing." << std::endl;
      exit(EXIT_FAILURE);
    } else {
      for (auto node_pop: pops) {
        std::size_t key = node_pop.first;
        ofs << key << " " << free_energies[key] << " " << node_pop.second << "\n";
      }
    }
  }

  std::set<std::size_t>
  compute_and_save_leaves(std::string fname,
                          std::map<std::size_t, std::size_t> network) {
    Clustering::logger(std::cout) << "saving leaves, i.e. tree end nodes" << std::endl;
    std::set<std::size_t> leaves;
    std::set<std::size_t> not_leaves;
    std::ofstream ofs(fname);
    if (ofs.fail()) {
      std::cerr << "error: cannot open file '" << fname << "' for writing." << std::endl;
      exit(EXIT_FAILURE);
    } else {
      for (auto from_to: network) {
        std::size_t src = from_to.first;
        std::size_t target = from_to.second;
        not_leaves.insert(target);
        if (not_leaves.count(src)) {
          leaves.erase(src);
        } else {
          leaves.insert(src);
        }
      }
      for (std::size_t leaf: leaves) {
        ofs << leaf << "\n";
      }
    }
    return leaves;
  }

  void
  save_traj_of_leaves(std::string fname,
                      std::set<std::size_t> leaves,
                      float d_min,
                      float d_max,
                      float d_step,
                      std::string remapped_name,
                      std::size_t n_rows) {
    Clustering::logger(std::cout) << "saving end-node trajectory for seeding" << std::endl;
    std::ofstream ofs(fname);
    if (ofs.fail()) {
      std::cerr << "error: cannot open file '" << fname << "' for writing." << std::endl;
      exit(EXIT_FAILURE);
    } else {
      std::vector<std::size_t> traj(n_rows);
      const float prec = d_step / 10.0f;
      for (float d=d_min; ! fuzzy_equal(d, d_max+d_step, prec); d += d_step) {
        std::vector<std::size_t> cl_now = Clustering::Tools::read_clustered_trajectory(
                                            Clustering::Tools::stringprintf(remapped_name, d));
        for (std::size_t i=0; i < n_rows; ++i) {
          if (leaves.count(cl_now[i])) {
            traj[i] = cl_now[i];
          }
        }
      }
      for (std::size_t i: traj) {
        ofs << i << "\n";
      }
    }
  }
  
  void
  save_network_to_html(std::string fname,
                       std::map<std::size_t, std::size_t> network,
                       std::map<std::size_t, float> free_energies,
                       std::map<std::size_t, std::size_t> pops) {
    Clustering::logger(std::cout) << "computing network visualization" << std::endl;
    // set (global) values for min/max of free energies and populations
    FE_MAX = std::max_element(free_energies.begin(),
                              free_energies.end(),
                              [](std::pair<std::size_t, float> fe1,
                                 std::pair<std::size_t, float> fe2) -> bool {
                                return fe1.second < fe2.second;
                              })->second;
    FE_MIN = std::min_element(free_energies.begin(),
                              free_energies.end(),
                              [](std::pair<std::size_t, float> fe1,
                                 std::pair<std::size_t, float> fe2) -> bool {
                                return fe1.second < fe2.second;
                              })->second;
    POP_MAX = std::max_element(pops.begin(),
                               pops.end(),
                               [](std::pair<std::size_t, std::size_t> p1,
                                  std::pair<std::size_t, std::size_t> p2) -> bool {
                                 return p1.second < p2.second;
                               })->second;
    POP_MIN = std::min_element(pops.begin(),
                               pops.end(),
                               [](std::pair<std::size_t, std::size_t> p1,
                                  std::pair<std::size_t, std::size_t> p2) -> bool {
                                 return p1.second < p2.second;
                               })->second;
    // build trees from given network with respective 'root' on top and at highest FE.
    // may be multiple trees because there may be multiple nodes that have max FE.
    Node fake_root;
    std::size_t network_size = network.size();
    std::size_t i_frame = 0;
    for (auto from_to: network) {
      std::cerr << ++i_frame << " / " << network_size << "\n";

      std::size_t i_from = from_to.first;
      std::size_t i_to = from_to.second;

      Node* parent_to = fake_root.find_parent_of(i_to);
      if ( ! parent_to) {
        fake_root.children[i_to] = {i_to, free_energies[i_to], pops[i_to]};
        parent_to = &fake_root;
      }
      Node* parent_from = fake_root.find_parent_of(i_from);
      if (parent_from) {
        // move existing node to its right place in the tree
        parent_to->children[i_to].children[i_from] = parent_from->children[i_from];
        parent_from->children.erase(i_from);
      } else {
        // create child at proper place in tree
        parent_to->children[i_to].children[i_from] = {i_from, free_energies[i_from], pops[i_from]};
      }
    }
    // write header
    std::ofstream ofs(fname);
    if (ofs.fail()) {
      std::cerr << "error: cannot open file '" << fname << "' for writing." << std::endl;
      exit(EXIT_FAILURE);
    } else {
      float LOG_POP_MIN, LOG_POP_MAX;
      if (POP_MIN <= 0) {
        LOG_POP_MIN = 0.0f;
      } else {
        LOG_POP_MIN = log(POP_MIN);
      }
      if (POP_MAX <= 0) {
        LOG_POP_MAX = 0.0f;
      } else {
        LOG_POP_MAX = log(POP_MAX);
      }
      ofs << Clustering::Network::viewer_header

          << "style: cytoscape.stylesheet().selector('node').css({"
          << Clustering::Tools::stringprintf("'width': 'mapData(logpop, %0.2f, %0.2f, 5, 30)',", LOG_POP_MIN, LOG_POP_MAX)
          << Clustering::Tools::stringprintf("'height': 'mapData(logpop, %0.2f, %0.2f, 5, 30)',", LOG_POP_MIN, LOG_POP_MAX)
          << Clustering::Tools::stringprintf("'background-color': 'mapData(fe, %f, %f, blue, red)'})", FE_MIN, FE_MAX)

          << ".selector('edge').css({'opacity': '1.0', 'width': '5', 'target-arrow-shape': 'triangle'})"
          << ".selector(':selected').css({'content': 'data(info)', 'font-size': 24, 'color': '#00ff00'})"

          << ", elements: [\n";
      fake_root.set_pos(0, 0);
      fake_root.print_subtree(ofs);
      ofs << "]";
      ofs << Clustering::Network::viewer_footer;
    }
  }
} // end local namespace


int main(int argc, char* argv[]) {
  using namespace Clustering::Tools;
  namespace b_po = boost::program_options;
  b_po::variables_map args;
  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "build network information from density based clustering."
    "\n"
    "options"));
  desc.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    // optional
    ("basename,b", b_po::value<std::string>()->default_value("clust.\%0.1f"),
          "(optional): basename of input files (default: clust.\%0.1f).")
    ("min", b_po::value<float>()->default_value(0.1f, "0.1"), "(optional): minimum free energy (default: 0.1).")
    ("max", b_po::value<float>()->default_value(8.0f, "8.0"), "(optional): maximum free energy (default: 8.0).")
    ("step", b_po::value<float>()->default_value(0.1f, "0.1"), "(optional): minimum free energy (default: 0.1).")
    ("minpop,p", b_po::value<std::size_t>()->default_value(1),
          "(optional): minimum population of node to be considered for network (default: 1).")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false), "verbose mode: print runtime information to STDOUT.")
  ;
  // parse cmd arguments
  try {
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).run(), args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cout << "\n" << e.what() << "\n\n" << std::endl;
    }
    std::cout << desc << std::endl;
    return 2;
  }
  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return 1;
  }
  // setup two threads parallel read/write
  omp_set_num_threads(2);
  // setup general flags / options
  Clustering::verbose = args["verbose"].as<bool>();

  float d_min = args["min"].as<float>();
  float d_max = args["max"].as<float>();
  float d_step = args["step"].as<float>();
  std::string basename = args["basename"].as<std::string>();
  std::string remapped_name = "remapped_" + basename;
  std::size_t minpop = args["minpop"].as<std::size_t>();

  std::map<std::size_t, std::size_t> network;
  std::map<std::size_t, std::size_t> pops;
  std::map<std::size_t, float> free_energies;

  std::vector<std::size_t> cl_next = read_clustered_trajectory(stringprintf(basename, d_min));
  std::vector<std::size_t> cl_now;
  std::size_t max_id;
  std::size_t n_rows = cl_next.size();
  // re-map states to give every state a unique id.
  // this is nevessary, since every initially clustered trajectory
  // at different thresholds uses the same ids starting with 0.
  const float prec = d_step / 10.0f;
  for (float d=d_min; ! fuzzy_equal(d, d_max, prec); d += d_step) {
    Clustering::logger(std::cout) << "free energy level: " << stringprintf("%0.2f", d) << std::endl;
    cl_now = cl_next;
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        write_clustered_trajectory(stringprintf(remapped_name, d), cl_now);
      }
      #pragma omp section
      {
        cl_next = read_clustered_trajectory(stringprintf(basename, d + d_step));
        max_id = *std::max_element(cl_now.begin(), cl_now.end());
        //TODO bugfix: correct pop count!
        for (std::size_t i=0; i < n_rows; ++i) {
          if (cl_next[i] != 0) {
            cl_next[i] += max_id;
            if (cl_now[i] != 0) {
              network[cl_now[i]] = cl_next[i];
              ++pops[cl_now[i]];
              free_energies[cl_now[i]] = d;
            }
          }
        }
      }
    }
  }
  // handle last trajectory
  Clustering::logger(std::cout) << "free energy level: " << stringprintf("%0.2f", d_max) << std::endl;
  cl_now = cl_next;
  write_clustered_trajectory(stringprintf(remapped_name, d_max), cl_now);
  for (std::size_t i=0; i < n_rows; ++i) {
    if (cl_now[i] != 0) {
      ++pops[cl_now[i]];
      free_energies[cl_now[i]] = d_max;
    }
  }
  // if minpop given: delete nodes and edges not fulfilling min. population criterium
  if (minpop > 1) {
    Clustering::logger(std::cout) << "cleaning from low pop. states ..." << std::endl;
    std::unordered_set<std::size_t> removals;
    auto pop_it = pops.begin();
    Clustering::logger(std::cout) << "  ... search nodes to remove" << std::endl;
    while (pop_it != pops.end()) {
      if (pop_it->second < minpop) {
        removals.insert(pop_it->first);
        pops.erase(pop_it++); // as above
      } else {
        ++pop_it;
      }
    }
    Clustering::logger(std::cout) << "  ... search edges to remove" << std::endl;
    auto net_it = network.begin();
    while (net_it != network.end()) {
      std::size_t a = net_it->first;
      std::size_t b = net_it->second;
      if (removals.count(a) > 0 || removals.count(b) > 0) {
        network.erase(net_it++);
      } else {
        ++net_it;
      }
    }
    Clustering::logger(std::cout) << "  ... finished." << std::endl;
  }
  // save links (i.e. edges) as two-column ascii in a child -> parent manner
  save_network_links("network_links.dat", network);
  // save id, population and free energy of nodes
  save_node_info("network_nodes.dat", free_energies, pops);
  // compute and directly save network end-nodes (i.e. leaves of the network-tree)
  std::set<std::size_t> leaves = compute_and_save_leaves("network_leaves.dat", network);
  // save the trajectory consisting of the 'leaf-states'.
  // all non-leaf states are kept as non-assignment state '0'.
  save_traj_of_leaves("network_end_node_traj.dat", leaves, d_min, d_max, d_step, remapped_name, n_rows);
  // generate html-file with embedded javascript to visualize network
  //TODO html-generation
  save_network_to_html("network_visualization.html", network, free_energies, pops);
  return 0;
}

