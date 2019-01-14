/*
Copyright (c) 2015-2019, Florian Sittel (www.lettis.net) and Daniel Nagel
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "network_builder.hpp"

#include "tools.hpp"
#include "logger.hpp"
#include "embedded_cytoscape.hpp"

#include <fstream>
#include <set>
#include <unordered_set>
#include <limits>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <omp.h>


namespace {
  //// use these values to construct
  //// the graphical network with cytoscape.js.
  //// in units of pixels.
  int HORIZONTAL_SPACING = 10;
  int VERTICAL_SPACING = 50;
//  int NODE_SIZE_MIN = 5;
//  int NODE_SIZE_MAX = 50;
  // the following values will be set by
  // 'save_network_to_html'
  // and are stored here for convenience.
  // needed by 'Node'-instances for
  // construction of graphical network,
  // i.e. to determine node sizes and colors.
  std::size_t POP_MIN = 0;
  std::size_t POP_MAX = 0;
  float       FE_MIN = 0.0f;
  float       FE_MAX = 0.0f;
  ////////

  struct Node {
    std::size_t id;
    float fe;
    std::size_t pop;
    std::map<std::size_t, Node> children;
    int pos_x = 0;
    int pos_y = 0;
    int _subtree_width = 0;

    Node();
    Node(std::size_t _id, float _fe, std::size_t _pop);
    Node* find_parent_of(std::size_t search_id);
    void set_pos(int x, int y);
    int subtree_width();
    void print_subtree(std::ostream& os);
    void print_node_and_subtree(std::ostream& os);
  };

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
  save_network_links(std::string fname, std::map<std::size_t, std::size_t> network,
                     std::string header_comment, std::map<std::string,float> commentsMap) {
    fname.append("_links.dat");
    Clustering::logger(std::cout) << "    saving links in: " << fname << std::endl;
    Clustering::Tools::append_commentsMap(header_comment, commentsMap);
    header_comment.append("#\n# Name of the cluster connected to the name in next "
                          "higher free energy level\n# Named by the remapped clusters.\n#\n"
                          "# cluster_name(fe+step) cluster_name(fe)\n");
    Clustering::Tools::write_map<std::size_t, std::size_t>(fname, network, header_comment, true);
  }
  
  void
  save_node_info(std::string fname,
                 std::map<std::size_t, float> free_energies,
                 std::map<std::size_t, std::size_t> pops,
                 std::string header_comment,
                 std::map<std::string,float> commentsMap) {
    fname.append("_nodes.dat");
    Clustering::logger(std::cout) << "    saving nodes in: " << fname << std::endl;
    Clustering::Tools::append_commentsMap(header_comment, commentsMap);
    header_comment.append("#\n# nodes\n");
    header_comment.append("#\n# Name of all clusters at a given free energies (fe) "
                          "with the corresponding populations pop.\n"
                          "# id(cluster) fe pop\n");
    std::ofstream ofs(fname);
    if (ofs.fail()) {
      std::cerr << "error: cannot open file '" << fname << "' for writing." << std::endl;
      exit(EXIT_FAILURE);
    } else {
      ofs << header_comment;
      for (auto node_pop: pops) {
        std::size_t key = node_pop.first;
        ofs << key << " " << free_energies[key] << " " << node_pop.second << "\n";
      }
    }
  }

  std::set<std::size_t>
  compute_and_save_leaves(std::string fname,
                          std::map<std::size_t, std::size_t> network,
                          std::string header_comment,
                          std::map<std::string,float> commentsMap) {
    fname.append("_leaves.dat");
    Clustering::logger(std::cout) << "    saving leaves in: " << fname << std::endl;
    std::set<std::size_t> leaves;
    std::set<std::size_t> not_leaves;
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
    std::vector<std::size_t> leaves_vec( leaves.begin(), leaves.end() );
    Clustering::Tools::append_commentsMap(header_comment, commentsMap);
    header_comment.append("#\n# All network leaves, i.e. nodes (microstates) without child\n"
                          "# nodes at a lower free energy level. These microstates represent\n"
                          "# the minima of their local basins.\n#\n"
                          "# id(cluster)\n"
                          );
    Clustering::Tools::write_single_column<std::size_t>(fname, leaves_vec, header_comment, false);
    return leaves;
  }

  void
  save_traj_of_leaves(std::string fname,
                      std::set<std::size_t> leaves,
                      float d_min,
                      float d_max,
                      float d_step,
                      std::string remapped_name,
                      std::size_t n_rows,
                      std::string header_comment,
                      std::map<std::string,float> commentsMap) {
    fname.append("_end_node_traj.dat");
    Clustering::logger(std::cout) << "    saving end-node trajectory in: " << fname << std::endl;
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
    Clustering::Tools::append_commentsMap(header_comment, commentsMap);
    header_comment.append("#\n# All frames beloning to a leaf node are marked with\n"
                          "# the custer id. All others with zero.\n");
    header_comment.append("#\n# state/cluster id frames are assigned to\n");
    Clustering::Tools::write_single_column<std::size_t>(fname, traj, header_comment);
  }
  
  void
  save_network_to_html(std::string fname,
                       std::map<std::size_t, std::size_t> network,
                       std::map<std::size_t, float> free_energies,
                       std::map<std::size_t, std::size_t> pops) {
    Clustering::logger(std::cout) << "\n~~~ computing network visualization" << std::endl;
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
    boost::progress_display show_progress(network_size);
    for (auto from_to: network) {
      ++show_progress;
      std::string ws(60, ' ');

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
    Clustering::logger(std::cout) << "    ...done" << std::endl;
    // write header
    fname.append("_visualization.html");
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


namespace Clustering {
namespace NetworkBuilder {

  void
  main(boost::program_options::variables_map args) {
    namespace b_fs = boost::filesystem;
    using namespace Clustering::Tools;
    // setup two threads parallel read/write
    omp_set_num_threads(2);
    // setup general flags / options
    Clustering::verbose = args["verbose"].as<bool>();

    float d_min = args["min"].as<float>();
    float d_max = args["max"].as<float>();
    float d_step = args["step"].as<float>();
    std::string basename = args["basename"].as<std::string>();
    std::string basename_output = args["output"].as<std::string>();
    basename.append(".%0.2f");
    std::string remapped_name = "remapped_" + basename;
    std::size_t minpop = args["minpop"].as<std::size_t>();
    bool network_html = args["network-html"].as<bool>();
    std::string header_comment = args["header"].as<std::string>();
    std::map<std::string,float> commentsMap = args["commentsMap"].as<std::map<std::string,float>>();

    std::map<std::size_t, std::size_t> network;
    std::map<std::size_t, std::size_t> pops;
    std::map<std::size_t, float> free_energies;


    std::string fname_next = stringprintf(basename, d_min);
    if ( ! b_fs::exists(fname_next)) {
      std::cerr << "error: file does not exist: " << fname_next
                << "       check basename (-b) and --min/--max/--step" << std::endl;
      exit(EXIT_SUCCESS);
    }
    read_comments(fname_next, commentsMap);
    std::vector<std::size_t> cl_next = read_clustered_trajectory(fname_next);
    std::vector<std::size_t> cl_now;
    std::size_t max_id;
    std::size_t n_rows = cl_next.size();
    // re-map states to give every state a unique id.
    // this is necessary, since every initially clustered trajectory
    // at different thresholds uses the same ids starting with 0.
    const float prec = d_step / 10.0f;
    if (d_max == 0.0f) {
      // default: collect all until MAX_FE
      d_max = std::numeric_limits<float>::max();
    } else {
      d_max += d_step;
    }
    float d;
    Clustering::logger(std::cout) << "~~~ remapping cluster files and generating network" << std::endl;
    for (d=d_min; ! fuzzy_equal(d, d_max, prec) && b_fs::exists(fname_next); d += d_step) {
      Clustering::logger(std::cout) << "    " << fname_next << " -> "
                                    << stringprintf(remapped_name, d)<< std::endl;
      cl_now = cl_next;
      fname_next = stringprintf(basename, d + d_step);
      #pragma omp parallel sections
      {
        #pragma omp section
        {
          write_clustered_trajectory(stringprintf(remapped_name, d),
                                     cl_now,
                                     header_comment,
                                     commentsMap);
        }
        #pragma omp section
        {
          if (b_fs::exists(fname_next)) {
            cl_next = read_clustered_trajectory(fname_next);
            max_id = *std::max_element(cl_now.begin(), cl_now.end());
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
    }
    // set correct value for d_max for later reference
    d_max = d-d_step;
    // if minpop given: delete nodes and edges not fulfilling min. population criterium
    commentsMap["minimal_population"] = minpop;
    if (minpop > 1) {
      Clustering::logger(std::cout) << "\n~~~ removing states with population p < "
                                    << minpop << std::endl;
      std::unordered_set<std::size_t> removals;
      auto pop_it = pops.begin();
      Clustering::logger(std::cout) << "    ... removing nodes" << std::endl;
      while (pop_it != pops.end()) {
        if (pop_it->second < minpop) {
          removals.insert(pop_it->first);
          pops.erase(pop_it++); // as above
        } else {
          ++pop_it;
        }
      }
      Clustering::logger(std::cout) << "    ... removing edges" << std::endl;
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
    }
    Clustering::logger(std::cout) << "\n~~~ storing output files" << std::endl;
    // save links (i.e. edges) as two-column ascii in a child -> parent manner
    save_network_links(basename_output, network, header_comment, commentsMap);
    // save id, population and free energy of nodes
    save_node_info(basename_output, free_energies, pops, header_comment, commentsMap);
    // compute and directly save network end-nodes (i.e. leaves of the network-tree)
    std::set<std::size_t> leaves = compute_and_save_leaves(basename_output,
                                                           network, header_comment, commentsMap);
    // save the trajectory consisting of the 'leaf-states'.
    // all non-leaf states are kept as non-assignment state '0'.
    save_traj_of_leaves(basename_output, leaves,
                        d_min, d_max, d_step, remapped_name, n_rows, header_comment, commentsMap);
    // generate html-file with embedded javascript to visualize network
    if (network_html) {
      save_network_to_html(basename_output, network, free_energies, pops);
    }
  }
} // end namespace NetworkBuilder
} // end namespace Clustering

