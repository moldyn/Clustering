#pragma once

#include <list>
#include <iostream>

namespace {

  //// use these values to construct
  //// the graphical network with cytoscape.js.
  //// in units of pixels.
  int HORIZONTAL_SPACING = 2;
  int VERTICAL_SPACING = 10;
  int NODE_SIZE_MIN = 5;
  int NODE_SIZE_MAX = 50;
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
    std::list<Node> children;
    int pos_x = 0;
    int pos_y = 0;
    int _subtree_width = 0;

    Node();
    Node(std::size_t _id, float _fe, std::size_t _pop);
    Node* find_child(std::size_t search_id);
    void set_pos(int x, int y);
    int subtree_width();
    void print_node_and_subtree(std::ostream& os);
  };

}; // end local namespace

