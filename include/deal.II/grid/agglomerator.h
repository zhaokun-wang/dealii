// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------


#ifndef agglomerator_h
#define agglomerator_h


#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/strategies.hpp>

template <int dim, int spacedim>
class AgglomerationHandler;

namespace dealii
{
  namespace internal
  {
    /**
     * This struct implements the R-tree agglomeration based on `boost::geometry::index::detail::rtree::visitor` 
     * interface. It traverses the spatial tree down to a specified `target_level`. 
     * Once it reaches the target depth, it groups all underlying leaf nodes (which 
     * correspond to standard deal.II active cells) into polytopes.
     */
    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    struct Rtree_visitor
      : public boost::geometry::index::detail::rtree::visitor<
          Value,
          typename Options::parameters_type,
          Box,
          Allocators,
          typename Options::node_tag,
          true>::type
    {
      inline Rtree_visitor(
        const Translator  &translator,
        const unsigned int target_level,
        std::vector<std::vector<typename Triangulation<
          boost::geometry::dimension<Box>::value>::active_cell_iterator>>
                                                        &agglomerates_,
        std::vector<types::global_cell_index>           &n_nodes_per_level/*,
        std::map<std::pair<types::global_cell_index, types::global_cell_index>,
                 std::vector<types::global_cell_index>> &parent_to_children*/);

      /**
       * An alias that identifies an InternalNode of the tree.
       */
      using InternalNode =
        typename boost::geometry::index::detail::rtree::internal_node<
          Value,
          typename Options::parameters_type,
          Box,
          Allocators,
          typename Options::node_tag>::type;

      /**
       * An alias that identifies a Leaf of the tree.
       */
      using Leaf = typename boost::geometry::index::detail::rtree::leaf<
        Value,
        typename Options::parameters_type,
        Box,
        Allocators,
        typename Options::node_tag>::type;

      /**
       * Implements the visitor interface for InternalNode objects. If the node
       * belongs to the level next to @p target_level, then fill the bounding
       * box vector for that node.
       */
      inline void
      operator()(const InternalNode &node);

      /**
       * Implements the visitor interface for Leaf objects.
       */
      inline void
      operator()(const Leaf &);

      /**
       * Translator interface, required by the boost implementation of the
       * rtree.
       */
      const Translator &translator;

      /**
       * Store the level we are currently visiting.
       */
      size_t level;

      /**
       * Index used to keep track of the number of different visited nodes
       * during recursion/
       */
      size_t node_counter;

      /**
       * The specific depth level within the R-tree to be extracted.
       * All active cells that belong to the same R-tree node at this depth level
       * are agglomerated into a single polytope.
       */
      const size_t target_level;

      /**
       * A reference to the input vector of vector of BoundingBox objects. This
       * vector v has the following property: v[i] = vector with all the mesh
       * iterators composing the i-th agglomerate.
       */
      std::vector<std::vector<typename Triangulation<
        boost::geometry::dimension<Box>::value>::active_cell_iterator>>
        &agglomerates;

      /**
       * Store the total number of nodes on each level.
       */
      std::vector<types::global_cell_index> &n_nodes_per_level;

      /**
       * Map that associates to a given node on level l its children, identified
       * by their integer index.
       */
      /*std::map<std::pair<types::global_cell_index, types::global_cell_index>,
               std::vector<types::global_cell_index>>
        &parent_node_to_children_nodes;*/
    };



    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    Rtree_visitor<Value, Options, Translator, Box, Allocators>::Rtree_visitor(
      const Translator  &translator,
      const unsigned int target_level,
      std::vector<std::vector<typename Triangulation<
        boost::geometry::dimension<Box>::value>::active_cell_iterator>>
                                                      &agglomerates_,
      std::vector<types::global_cell_index>           &n_nodes_per_level_/*,
      std::map<std::pair<types::global_cell_index, types::global_cell_index>,
               std::vector<types::global_cell_index>> &parent_to_children*/)
      : translator(translator)
      , level(0)
      , node_counter(0)
      , target_level(target_level)
      , agglomerates(agglomerates_)
      , n_nodes_per_level(n_nodes_per_level_)
      //, parent_node_to_children_nodes(parent_to_children)
    {}



    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    void
    Rtree_visitor<Value, Options, Translator, Box, Allocators>::operator()(
      const Rtree_visitor::InternalNode &node)
    {
      using elements_type =
        typename boost::geometry::index::detail::rtree::elements_type<
          InternalNode>::type; //  pairs of bounding box and pointer to child
                               //  node
      const elements_type &elements =
        boost::geometry::index::detail::rtree::elements(node);

      if (level < target_level)
        {
          size_t level_backup = level;
          ++level;

          for (typename elements_type::const_iterator it = elements.begin();
               it != elements.end();
               ++it)
            {
              boost::geometry::index::detail::rtree::apply_visitor(*this,
                                                                   *it->second);
            }

          level = level_backup;
        }
      else if (level == target_level)
        {
          const auto offset = agglomerates.size();
          agglomerates.resize(offset + 1);
          size_t level_backup = level;

          ++level;
          for (const auto &entry : elements)
            {
              boost::geometry::index::detail::rtree::apply_visitor(
                *this, *entry.second);
            }
          // Done with node number 'node_counter' on level target_level.

          ++node_counter; // visited all children of an internal node
          n_nodes_per_level[target_level]++;

          level = level_backup;
        }
      else if (level > target_level)
        {
          // I am on a child (internal) node on a deeper level.

          // Keep visiting until you go to the leafs.
          size_t level_backup = level;

          ++level;

          // looping through entries of node
          for (const auto &entry : elements)
            {
              boost::geometry::index::detail::rtree::apply_visitor(
                *this, *entry.second);
            }
          // done with node on level l > target_level (not just
          // "target_level+1).
          n_nodes_per_level[level_backup]++;
          const types::global_cell_index node_idx =
            n_nodes_per_level[level_backup] - 1; // so to start from 0

          /*parent_node_to_children_nodes[{n_nodes_per_level[level_backup - 1],
                                         level_backup - 1}]
            .push_back(node_idx);*/

          level = level_backup;
        }
    }



    template <typename Value,
              typename Options,
              typename Translator,
              typename Box,
              typename Allocators>
    void
    Rtree_visitor<Value, Options, Translator, Box, Allocators>::operator()(
      const Rtree_visitor::Leaf &leaf)
    {
      using elements_type =
        typename boost::geometry::index::detail::rtree::elements_type<
          Leaf>::type; //  pairs of bounding box and pointer to child node
      const elements_type &elements =
        boost::geometry::index::detail::rtree::elements(leaf);

      if (level == target_level)
        {
          // If I want to extract from leaf node, i.e. the target_level is the
          // last one where leafs are grouped together.
          const auto offset = agglomerates.size();
          agglomerates.resize(offset + 1);

          for (const auto &it : elements)
            agglomerates[node_counter].push_back(it.second);

          ++node_counter;
          n_nodes_per_level[target_level]++;
        }
      else
        {
          for (const auto &it : elements)
            agglomerates[node_counter].push_back(it.second);


          if (level == target_level + 1)
            {
              const unsigned int node_idx = n_nodes_per_level[level];

              /*parent_node_to_children_nodes[{n_nodes_per_level[level - 1],
                                             level - 1}]
                .push_back(node_idx);*/
              n_nodes_per_level[level]++;
            }
        }
    }
  } // namespace internal



  /**
   * Helper class which handles agglomeration based on the R-tree data
   * structure. Notice that the R-tree type is assumed to be an R-star-tree.
   */
  template <int dim, typename RtreeType>
  class CellsAgglomerator
  {
  public:
    template <int, int>
    friend class ::AgglomerationHandler;

    /**
     * Constructor. It takes a given rtree and an integer representing the
     * index of the level to be extracted.
     */
    CellsAgglomerator(const RtreeType   &rtree,
                      const unsigned int extraction_level);

    /**
     * Extract agglomerates based on the current tree and the extraction level.
     * It groups all active cells that belong to the same R-tree node at the specified 
     * @p target_level into a single agglomerated polytope.
     */
    const std::vector<
      std::vector<typename Triangulation<dim>::active_cell_iterator>> &
    extract_agglomerates();

    /**
     * Get total number of levels.
     */
    inline unsigned int
    get_n_levels() const;

    /**
     * Return the number of nodes present in level @p level.
     */
    inline types::global_cell_index
    get_n_nodes_per_level(const unsigned int level) const;

    /**
     * This function returns a map which associates to each node on level
     * @p extraction_level a list of children.
     */
    /*inline const std::map<
      std::pair<types::global_cell_index, types::global_cell_index>,
      std::vector<types::global_cell_index>> &
    get_hierarchy() const;*/

  private:
    /**
     * Raw pointer to the actual R-tree.
     */
    RtreeType *rtree;

    /**
     * Extraction level.
     */
    const unsigned int extraction_level;

    /**
     * Store agglomerates obtained after recursive extraction on nodes of
     * level @p extraction_level.
     */
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      agglomerates_on_level;

    /**
     * Vector storing the number of nodes (and, ultimately, agglomerates) for
     * each level.
     */
    std::vector<types::global_cell_index> n_nodes_per_level;

    /**
     * Map which maps a node parent @n on level @p l to a vector of integers
     * which stores the index of children.
     */
    /*std::map<std::pair<types::global_cell_index, types::global_cell_index>,
             std::vector<types::global_cell_index>>
      parent_node_to_children_nodes;*/
  };



  template <int dim, typename RtreeType>
  CellsAgglomerator<dim, RtreeType>::CellsAgglomerator(
    const RtreeType   &tree,
    const unsigned int extraction_level_)
    : extraction_level(extraction_level_)
  {
    rtree = const_cast<RtreeType *>(&tree);
    Assert(n_levels(*rtree), ExcMessage("At least two levels are needed."));
  }



  template <int dim, typename RtreeType>
  const std::vector<
    std::vector<typename Triangulation<dim>::active_cell_iterator>> &
  CellsAgglomerator<dim, RtreeType>::extract_agglomerates()
  {
    AssertThrow(extraction_level <= n_levels(*rtree),
                ExcInternalError("You are trying to extract level " +
                                 std::to_string(extraction_level) +
                                 " of the tree, but it only has a total of " +
                                 std::to_string(n_levels(*rtree)) +
                                 " levels."));
    using RtreeView =
      boost::geometry::index::detail::rtree::utilities::view<RtreeType>;
    RtreeView rtv(*rtree);

    n_nodes_per_level.resize(rtv.depth() +
                             1); // store how many nodes we have for each level.

    if (rtv.depth() == 0)
      {
        // The below algorithm does not work for `rtv.depth()==0`, which might
        // happen if the number entries in the tree is too small.
        agglomerates_on_level.resize(1);
        agglomerates_on_level[0].resize(1);
      }
    else
      {
        const unsigned int target_level =
          std::min<unsigned int>(extraction_level, rtv.depth());

        internal::Rtree_visitor<typename RtreeView::value_type,
                                typename RtreeView::options_type,
                                typename RtreeView::translator_type,
                                typename RtreeView::box_type,
                                typename RtreeView::allocators_type>
          extractor_visitor(rtv.translator(),
                            target_level,
                            agglomerates_on_level,
                            n_nodes_per_level/*,
                            parent_node_to_children_nodes*/);


        rtv.apply_visitor(extractor_visitor);
      }
    return agglomerates_on_level;
  }

  // ------------------------------ inline functions -------------------------


  template <int dim, typename RtreeType>
  inline unsigned int
  CellsAgglomerator<dim, RtreeType>::get_n_levels() const
  {
    return n_levels(*rtree);
  }



  template <int dim, typename RtreeType>
  inline types::global_cell_index
  CellsAgglomerator<dim, RtreeType>::get_n_nodes_per_level(
    const unsigned int level) const
  {
    return n_nodes_per_level[level];
  }

  /*template <int dim, typename RtreeType>
  inline const std::map<
    std::pair<types::global_cell_index, types::global_cell_index>,
    std::vector<types::global_cell_index>> &
  CellsAgglomerator<dim, RtreeType>::get_hierarchy() const
  {
    Assert(parent_node_to_children_nodes.size(),
           ExcMessage(
             "The hierarchy has not been computed. Did you forget to call"
             " extract_agglomerates() first?"));
    return parent_node_to_children_nodes;
  }*/


  // ------------------------------ utility functions -------------------------

  namespace PolyUtils
  {
    namespace internal
    {
      /**
       * Helper function to compute the position of index @p index in vector @p v.
       */
      inline types::global_cell_index
      get_index(const std::vector<types::global_cell_index> &v,
                const types::global_cell_index               index)
      {
        return std::distance(v.begin(), std::find(v.begin(), v.end(), index));
      }



      /**
       * Compute the connectivity graph for locally owned regions of a distributed
       * triangulation.
       */
      template <int dim, int spacedim>
      void
      get_face_connectivity_of_cells(
        const parallel::fullydistributed::Triangulation<dim, spacedim>
                                                  &triangulation,
        DynamicSparsityPattern                     &cell_connectivity,
        const std::vector<types::global_cell_index> locally_owned_cells)
      {
        cell_connectivity.reinit(triangulation.n_locally_owned_active_cells(),
                                triangulation.n_locally_owned_active_cells());


        // loop over all cells and their neighbors to build the sparsity
        // pattern. note that it's a bit hard to enter all the connections when
        // a neighbor has children since we would need to find out which of its
        // children is adjacent to the current cell. this problem can be omitted
        // if we only do something if the neighbor has no children -- in that
        // case it is either on the same or a coarser level than we are. in
        // return, we have to add entries in both directions for both cells
        for (const auto &cell : triangulation.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                const unsigned int index = cell->active_cell_index();
                cell_connectivity.add(get_index(locally_owned_cells, index),
                                      get_index(locally_owned_cells, index));
                for (auto f : cell->face_indices())
                  if ((cell->at_boundary(f) == false) &&
                      (cell->neighbor(f)->has_children() == false) &&
                      cell->neighbor(f)->is_locally_owned())
                    {
                      const unsigned int other_index =
                        cell->neighbor(f)->active_cell_index();

                      cell_connectivity.add(get_index(locally_owned_cells, index),
                                            get_index(locally_owned_cells,
                                                      other_index));
                      cell_connectivity.add(get_index(locally_owned_cells,
                                                      other_index),
                                            get_index(locally_owned_cells, index));
                    }
              }
          }
      }
    }// namespace internal

    /**
     * Partition with METIS the locally owned regions of the given
     * triangulation.
     *
     * @note The given triangulation must be a parallel::fullydistributed::Triangulation. This is
     * required as the partitions generated by p4est, the partitioner for
     * parallell::distributed::Triangulation, which can generate discontinuous
     * partitions which are not supported by the METIS partitioner.
     *
     */
    template <int dim, int spacedim>
    void
    partition_locally_owned_regions(const unsigned int            n_partitions,
                                    Triangulation<dim, spacedim> &triangulation,
                                    const SparsityTools::Partitioner partitioner)
    {
      AssertDimension(dim, spacedim);
      Assert(n_partitions > 0,
            ExcMessage("Invalid number of partitions, you provided " +
                        std::to_string(n_partitions)));

      auto parallel_triangulation =
        dynamic_cast<parallel::fullydistributed::Triangulation<dim, spacedim> *>(
          &triangulation);
      Assert(
        (parallel_triangulation != nullptr),
        ExcMessage(
          "Only fully distributed triangulations are supported. If you are using"
          "a parallel::distributed::triangulation, you must convert it to a fully"
          "distributed as explained in the documentation."));

      // check for an easy return
      if (n_partitions == 1)
        {
          for (const auto &cell : parallel_triangulation->active_cell_iterators())
            if (cell->is_locally_owned())
              cell->set_material_id(0);
          return;
        }

      // collect all locally owned cells
      std::vector<types::global_cell_index> locally_owned_cells;
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          locally_owned_cells.push_back(cell->active_cell_index());

      DynamicSparsityPattern cell_connectivity;
      internal::get_face_connectivity_of_cells(*parallel_triangulation,
                                              cell_connectivity,
                                              locally_owned_cells);

      SparsityPattern sp_cell_connectivity;
      sp_cell_connectivity.copy_from(cell_connectivity);

      // partition each locally owned connection graph and get
      // back a vector of indices, one per degree
      // of freedom (which is associated with a
      // cell)
      std::vector<unsigned int> partition_indices(
        parallel_triangulation->n_locally_owned_active_cells());
      SparsityTools::partition(sp_cell_connectivity,
                              n_partitions,
                              partition_indices,
                              partitioner);


      // finally loop over all cells and set the material ids
      for (const auto &cell : parallel_triangulation->active_cell_iterators())
        if (cell->is_locally_owned())
          cell->set_material_id(
            partition_indices[internal::get_index(locally_owned_cells,
                                                  cell->active_cell_index())]);
    }

    /**
     * Agglomerate cells together based on their global index. This function is
     * **not** efficient and should be used for testing purposes only.
     */
    template <int dim, int spacedim = dim>
    void
    collect_cells_for_agglomeration(
      const Triangulation<dim, spacedim>          &tria,
      const std::vector<types::global_cell_index> &cell_idxs,
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
        &cells_to_be_agglomerated)
    {
      Assert(cells_to_be_agglomerated.size() == 0,
            ExcMessage(
              "The vector of cells is supposed to be filled by this function."));
      for (const auto &cell : tria.active_cell_iterators())
        if (std::find(cell_idxs.begin(),
                      cell_idxs.end(),
                      cell->active_cell_index()) != cell_idxs.end())
          {
            cells_to_be_agglomerated.push_back(cell);
          }
    }

    /**
     * This function traverses the given R-tree up to the requested @p level
     * and groups all the active cells on the underlying leaves into agglomerates
     * corresponds to the nodes on @p level .
     */
    template <typename Rtree>
    inline std::pair<
      // std::vector<std::vector<unsigned int>>,
      std::vector<unsigned int>,
      std::vector<std::vector<typename Triangulation<boost::geometry::dimension<
        typename Rtree::indexable_type>::value>::active_cell_iterator>>>
    extract_children_of_level(const Rtree &tree, const unsigned int level)
    {
      using RtreeView =
        boost::geometry::index::detail::rtree::utilities::view<Rtree>;
      RtreeView rtv(tree);

      // std::vector<std::vector<unsigned int>> csrs;
      std::vector<unsigned int> csrs;
      std::vector<std::vector<typename Triangulation<boost::geometry::dimension<
        typename Rtree::indexable_type>::value>::active_cell_iterator>>
        agglomerates;

      if (rtv.depth() == 0)
        {
          // The below algorithm does not work for `rtv.depth()==0`, which might
          // happen if the number entries in the tree is too small.
          // In this case, simply return a single bounding box.
          agglomerates.resize(1);
          agglomerates[0].resize(1);
          csrs.resize(1);
          // csrs[0].resize(1);
          csrs[0] = 1;
        }
      else
        {
          const unsigned int target_level =
            std::min<unsigned int>(level, rtv.depth());

          csrs.resize(rtv.depth() + 1, 0);

          dealii::internal::Rtree_visitor<
            typename RtreeView::value_type,
            typename RtreeView::options_type,
            typename RtreeView::translator_type,
            typename RtreeView::box_type,
            typename RtreeView::allocators_type>
            node_visitor(rtv.translator(), target_level, agglomerates, csrs);
          rtv.apply_visitor(node_visitor);
        }
      // AssertDimension(agglomerates.size(), csrs.size());

      return {csrs, agglomerates};
    }

    /**
     * Return a compile-time constant required for passing max_elem_per_node in R-tree initialization.
     */
    template <typename T>
    inline constexpr T
    constexpr_pow(T num, unsigned int pow)
    {
      return (pow >= sizeof(unsigned int) * 8) ? 0 :
            pow == 0                          ? 1 :
                                                num * constexpr_pow(num, pow - 1);
    }
  }// namespace PolyUtils
} // namespace dealii
#endif