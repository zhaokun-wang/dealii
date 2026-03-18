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
#ifndef agglomeration_handler_h
#define agglomeration_handler_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/cell_id.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/immersed_surface_quadrature.h>

#include <deal.II/numerics/vector_tools_common.h>

#include <deal.II/dofs/agglomeration_iterator.h>
#include <deal.II/grid/agglomerator.h>
#include <deal.II/fe/mapping_box.h>

#include <fstream>

using namespace dealii;

// Forward declarations
template <int dim, int spacedim>
class AgglomerationHandler;

namespace dealii
{
  namespace internal
  {
    /**
     * Helper class to reinit finite element spaces on polytopal cells.
     */
    template <int, int>
    class AgglomerationHandlerImplementation;
  } // namespace internal
} // namespace dealii



/**
 * Helper class for the storage of connectivity information of the polytopal
 * grid.
 */
namespace dealii
{
  namespace internal
  {
    template <int dim, int spacedim>
    class PolytopeCache
    {
    public:
      /**
       * Default constructor.
       */
      PolytopeCache() = default;

      /**
       * Destructor. It simply calls clear() for all of its members.
       */
      ~PolytopeCache() = default;

      void
      clear()
      {
        // clear all the members
        cell_face_at_boundary.clear();
        interface.clear();
        visited_cell_and_faces.clear();
      }

      /**
       * Standard std::set for recording the standard cells and faces (in the
       * deal.II lingo) that have been already visited. The first argument of
       * the pair identifies the global index of a deal.II cell, while the
       * second its local face number.
       *
       */
      mutable std::set<std::pair<types::global_cell_index, unsigned int>>
        visited_cell_and_faces;


      mutable std::set<std::pair<CellId, unsigned int>>
        visited_cell_and_faces_id;



      /**
       * Map that associate the pair of (polytopal index, polytopal face) to
       * (b,cell). The latter pair indicates whether or not the present face is
       * on boundary. If it's on the boundary, then b is true and cell is an
       * invalid cell iterator. Otherwise, b is false and cell points to the
       * neighboring polytopal cell.
       *
       */
      mutable std::map<
        std::pair<types::global_cell_index, unsigned int>,
        std::pair<bool,
                  typename Triangulation<dim, spacedim>::active_cell_iterator>>
        cell_face_at_boundary;

      /**
       * Map that associate the **local** pair of (polytope id, polytopal face)
       * to the master id of the neighboring **ghosted** cell.
       */
      mutable std::map<std::pair<CellId, unsigned int>, CellId>
        ghosted_master_id;

      /**
       * Standard std::map that associated to a pair of neighboring polytopic
       * cells (current_polytope, neighboring_polytope) a sequence of
       * ({deal_cell,deal_face_index}) which is meant to describe their
       * interface.
       * Indeed, the pair is identified by the two polytopic global indices,
       * while the interface is described by a std::vector of deal.II cells and
       * faces.
       *
       */
      mutable std::map<
        std::pair<CellId, CellId>,
        std::vector<
          std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
                    unsigned int>>>
        interface;
    };
  } // namespace internal
} // namespace dealii

namespace dealii::PolyUtils
{
  namespace internal
  {
    /**
     * Same as the public free function with the same name, but storing
     * explicitly the interpolation matrix and performing interpolation through
     * matrix-vector product.
     */
    template <int dim, int spacedim, typename VectorType>
    void
    interpolate_to_fine_grid(
      const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
      VectorType                                &dst,
      const VectorType                          &src)
    {
      Assert((dim == spacedim), ExcNotImplemented());
      Assert(
        dst.size() == 0,
        ExcMessage(
          "The destination vector must the empt upon calling this function."));

      using NumberType = typename VectorType::value_type;
      constexpr bool is_trilinos_vector =
        std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>;
      using MatrixType = std::conditional_t<is_trilinos_vector,
                                            TrilinosWrappers::SparseMatrix,
                                            SparseMatrix<NumberType>>;

      MatrixType interpolation_matrix;

      [[maybe_unused]]
      typename std::conditional_t<!is_trilinos_vector, SparsityPattern, void *>
        sp;

      // Get some info from the handler
      const DoFHandler<dim, spacedim> &agglo_dh =
        agglomeration_handler.agglo_dh;

      DoFHandler<dim, spacedim> *output_dh =
        const_cast<DoFHandler<dim, spacedim> *>(
          &agglomeration_handler.output_dh);
      const FiniteElement<dim, spacedim> &fe = agglomeration_handler.get_fe();
      const Mapping<dim> &mapping = agglomeration_handler.get_mapping();
      const Triangulation<dim, spacedim> &tria =
        agglomeration_handler.get_triangulation();
      const auto &bboxes = agglomeration_handler.get_local_bboxes();

      std::unique_ptr<FiniteElement<dim>> output_fe;
      if (tria.all_reference_cells_are_hyper_cube())
        output_fe = std::make_unique<FE_DGQ<dim>>(fe.degree);
      else if (tria.all_reference_cells_are_simplex())
        output_fe = std::make_unique<FE_SimplexDGP<dim>>(fe.degree);
      else
        AssertThrow(false, ExcNotImplemented());

      // Setup an auxiliary DoFHandler for output purposes
      output_dh->reinit(tria);
      output_dh->distribute_dofs(*output_fe);

      const IndexSet &locally_owned_dofs = output_dh->locally_owned_dofs();
      const IndexSet  locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(*output_dh);

      const IndexSet &locally_owned_dofs_agglo = agglo_dh.locally_owned_dofs();


      DynamicSparsityPattern dsp(output_dh->n_dofs(),
                                 agglo_dh.n_dofs(),
                                 locally_relevant_dofs);

      std::vector<types::global_dof_index> agglo_dof_indices(fe.dofs_per_cell);
      std::vector<types::global_dof_index> standard_dof_indices(
        fe.dofs_per_cell);
      std::vector<types::global_dof_index> output_dof_indices(
        output_fe->dofs_per_cell);

      Quadrature<dim>         quad(output_fe->get_unit_support_points());
      FEValues<dim, spacedim> output_fe_values(mapping,
                                               *output_fe,
                                               quad,
                                               update_quadrature_points);

      for (const auto &cell : agglo_dh.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (agglomeration_handler.is_master_cell(cell))
              {
                auto slaves = agglomeration_handler.get_slaves_of_idx(
                  cell->active_cell_index());
                slaves.emplace_back(cell);

                cell->get_dof_indices(agglo_dof_indices);

                for (const auto &slave : slaves)
                  {
                    // addd master-slave relationship
                    const auto slave_output =
                      slave->as_dof_handler_iterator(*output_dh);
                    slave_output->get_dof_indices(output_dof_indices);
                    for (const auto row : output_dof_indices)
                      dsp.add_entries(row,
                                      agglo_dof_indices.begin(),
                                      agglo_dof_indices.end());
                  }
              }
          }


      const auto assemble_interpolation_matrix = [&]() {
        FullMatrix<NumberType> local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
        std::vector<Point<dim>> reference_q_points(fe.dofs_per_cell);

        // Dummy AffineConstraints, only needed for loc2glb
        AffineConstraints<NumberType> c;
        c.close();

        for (const auto &cell : agglo_dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              if (agglomeration_handler.is_master_cell(cell))
                {
                  auto slaves = agglomeration_handler.get_slaves_of_idx(
                    cell->active_cell_index());
                  slaves.emplace_back(cell);

                  cell->get_dof_indices(agglo_dof_indices);

                  const types::global_cell_index polytope_index =
                    agglomeration_handler.cell_to_polytope_index(cell);

                  // Get the box of this agglomerate.
                  const BoundingBox<dim> &box = bboxes[polytope_index];

                  for (const auto &slave : slaves)
                    {
                      // add master-slave relationship
                      const auto slave_output =
                        slave->as_dof_handler_iterator(*output_dh);

                      slave_output->get_dof_indices(output_dof_indices);
                      output_fe_values.reinit(slave_output);

                      local_matrix = 0.;

                      const auto &q_points =
                        output_fe_values.get_quadrature_points();
                      for (const auto i : output_fe_values.dof_indices())
                        {
                          const auto &p = box.real_to_unit(q_points[i]);
                          for (const auto j : output_fe_values.dof_indices())
                            {
                              local_matrix(i, j) = fe.shape_value(j, p);
                            }
                        }
                      c.distribute_local_to_global(local_matrix,
                                                   output_dof_indices,
                                                   agglo_dof_indices,
                                                   interpolation_matrix);
                    }
                }
            }
      };


      if constexpr (std::is_same_v<MatrixType, TrilinosWrappers::SparseMatrix>)
        {
          const MPI_Comm &communicator = tria.get_communicator();
          SparsityTools::distribute_sparsity_pattern(dsp,
                                                     locally_owned_dofs,
                                                     communicator,
                                                     locally_relevant_dofs);

          interpolation_matrix.reinit(locally_owned_dofs,
                                      locally_owned_dofs_agglo,
                                      dsp,
                                      communicator);
          dst.reinit(locally_owned_dofs);
          assemble_interpolation_matrix();
        }
      else if constexpr (std::is_same_v<MatrixType, SparseMatrix<NumberType>>)
        {
          sp.copy_from(dsp);
          interpolation_matrix.reinit(sp);
          dst.reinit(output_dh->n_dofs());
          assemble_interpolation_matrix();
        }
      else
        {
          // PETSc, LA::d::v options not implemented.
          (void)agglomeration_handler;
          (void)dst;
          (void)src;
          AssertThrow(false, ExcNotImplemented());
        }

      // If tria is distributed
      if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &tria) != nullptr)
        interpolation_matrix.compress(VectorOperation::add);

      // Finally, perform the interpolation.
      interpolation_matrix.vmult(dst, src);
    }
  }

  /**
   * Given a vector @p src, typically the solution stemming after the
   * agglomerate problem has been solved, this function interpolates @p src
   * onto the finer grid and stores the result in vector @p dst. The last
   * argument @p on_the_fly does not build any interpolation matrix and allows
   * computing the entries in @p dst in a matrix-free fashion.
   *
   * @note Supported parallel types are TrilinosWrappers::SparseMatrix and
   * TrilinosWrappers::MPI::Vector.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  interpolate_to_fine_grid(
    const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
    VectorType                                &dst,
    const VectorType                          &src,
    const bool                                 on_the_fly = true)
  {
    Assert((dim == spacedim), ExcNotImplemented());
    Assert(
      dst.size() == 0,
      ExcMessage(
        "The destination vector must the empt upon calling this function."));

    using NumberType = typename VectorType::value_type;
    static constexpr bool is_trilinos_vector =
      std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>;

    static constexpr bool is_supported_vector =
      std::is_same_v<VectorType, Vector<NumberType>> || is_trilinos_vector;
    static_assert(is_supported_vector);

    // First, check for an easy return
    if (on_the_fly == false)
      {
        return internal::interpolate_to_fine_grid(agglomeration_handler,
                                                  dst,
                                                  src);
      }
    else
      {
        // otherwise, do not create any matrix
        const Triangulation<dim, spacedim> &tria =
          agglomeration_handler.get_triangulation();
        const Mapping<dim> &mapping = agglomeration_handler.get_mapping();
        const FiniteElement<dim, spacedim> &original_fe =
          agglomeration_handler.get_fe();

        // We use DGQ (on tensor-product meshes) or DGP (on simplex meshes)
        // nodal elements of the same degree as the ones in the agglomeration
        // handler to interpolate the solution onto the finer grid.
        std::unique_ptr<FiniteElement<dim>> output_fe;
        if (tria.all_reference_cells_are_hyper_cube())
          output_fe = std::make_unique<FE_DGQ<dim>>(original_fe.degree);
        else if (tria.all_reference_cells_are_simplex())
          output_fe = std::make_unique<FE_SimplexDGP<dim>>(original_fe.degree);
        else
          AssertThrow(false, ExcNotImplemented());

        DoFHandler<dim> &output_dh =
          const_cast<DoFHandler<dim> &>(agglomeration_handler.output_dh);
        output_dh.reinit(tria);
        output_dh.distribute_dofs(*output_fe);

        if constexpr (std::is_same_v<VectorType, TrilinosWrappers::MPI::Vector>)
          {
            const IndexSet &locally_owned_dofs = output_dh.locally_owned_dofs();
            dst.reinit(locally_owned_dofs);
          }
        else if constexpr (std::is_same_v<VectorType, Vector<NumberType>>)
          {
            dst.reinit(output_dh.n_dofs());
          }
        else
          {
            // PETSc, LA::d::v options not implemented.
            (void)agglomeration_handler;
            (void)dst;
            (void)src;
            AssertThrow(false, ExcNotImplemented());
          }



        const unsigned int dofs_per_cell =
          agglomeration_handler.n_dofs_per_cell();
        const unsigned int output_dofs_per_cell = output_fe->n_dofs_per_cell();
        Quadrature<dim>    quad(output_fe->get_unit_support_points());
        FEValues<dim>      output_fe_values(mapping,
                                       *output_fe,
                                       quad,
                                       update_quadrature_points);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices_output(
          output_dofs_per_cell);

        const auto &bboxes = agglomeration_handler.get_local_bboxes();
        for (const auto &polytope : agglomeration_handler.polytope_iterators())
          {
            if (polytope->is_locally_owned())
              {
                polytope->get_dof_indices(local_dof_indices);
                const BoundingBox<dim> &box = bboxes[polytope->index()];

                const auto &deal_cells =
                  polytope->get_agglomerate(); // fine deal.II cells
                for (const auto &cell : deal_cells)
                  {
                    const auto slave_output = cell->as_dof_handler_iterator(
                      agglomeration_handler.output_dh);
                    slave_output->get_dof_indices(local_dof_indices_output);
                    output_fe_values.reinit(slave_output);

                    const auto &qpoints =
                      output_fe_values.get_quadrature_points();

                    for (unsigned int j = 0; j < output_dofs_per_cell; ++j)
                      {
                        const auto &ref_qpoint = box.real_to_unit(qpoints[j]);
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          dst(local_dof_indices_output[j]) +=
                            src(local_dof_indices[i]) *
                            original_fe.shape_value(i, ref_qpoint);
                      }
                  }
              }
          }
      }
  }

  /**
   * Similar to VectorTools::compute_global_error(), but customized for
   * polytopic elements. Aside from the solution vector and a reference
   * function, this function takes in addition a vector @p norms with types
   * VectorTools::NormType to be computed and later stored in the last
   * argument @p global_errors.
   * In case of a parallel vector, the local errors are collected over each
   * processor and later a classical reduction operation is performed.
   */
  template <int dim, int spacedim, typename Number, typename VectorType>
  void
  compute_global_error(const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
                       const VectorType                &solution,
                       const Function<dim, Number>     &exact_solution,
                       const std::vector<VectorTools::NormType> &norms,
                       std::vector<double>                      &global_errors)
  {
    Assert(solution.size() > 0,
           ExcNotImplemented(
             "Solution vector must be non-empty upon calling this function."));
    Assert(std::any_of(norms.cbegin(),
                       norms.cend(),
                       [](VectorTools::NormType norm_type) {
                         return (norm_type ==
                                   VectorTools::NormType::H1_seminorm ||
                                 norm_type == VectorTools::NormType::L2_norm);
                       }),
           ExcMessage("Norm type not supported"));
    global_errors.resize(norms.size());
    std::fill(global_errors.begin(), global_errors.end(), 0.);

    // Vector storing errors local to the current processor.
    std::vector<double> local_errors(norms.size());
    std::fill(local_errors.begin(), local_errors.end(), 0.);

    // Get some info from the handler
    const unsigned int dofs_per_cell = agglomeration_handler.n_dofs_per_cell();

    const bool compute_semi_H1 =
      std::any_of(norms.cbegin(),
                  norms.cend(),
                  [](VectorTools::NormType norm_type) {
                    return norm_type == VectorTools::NormType::H1_seminorm;
                  });

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &polytope : agglomeration_handler.polytope_iterators())
      {
        if (polytope->is_locally_owned())
          {
            const auto &agglo_values = agglomeration_handler.reinit(polytope);
            polytope->get_dof_indices(local_dof_indices);

            const auto         &q_points = agglo_values.get_quadrature_points();
            const unsigned int  n_qpoints = q_points.size();
            std::vector<double> analyical_sol_at_qpoints(n_qpoints);
            exact_solution.value_list(q_points, analyical_sol_at_qpoints);
            std::vector<Tensor<1, dim>> grad_analyical_sol_at_qpoints(
              n_qpoints);

            if (compute_semi_H1)
              exact_solution.gradient_list(q_points,
                                           grad_analyical_sol_at_qpoints);

            for (unsigned int q_index : agglo_values.quadrature_point_indices())
              {
                double         solution_at_qpoint = 0.;
                Tensor<1, dim> grad_solution_at_qpoint;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    solution_at_qpoint += solution(local_dof_indices[i]) *
                                          agglo_values.shape_value(i, q_index);

                    if (compute_semi_H1)
                      grad_solution_at_qpoint +=
                        solution(local_dof_indices[i]) *
                        agglo_values.shape_grad(i, q_index);
                  }
                // L2
                local_errors[0] += std::pow((analyical_sol_at_qpoints[q_index] -
                                             solution_at_qpoint),
                                            2) *
                                   agglo_values.JxW(q_index);

                // H1 seminorm
                if (compute_semi_H1)
                  for (unsigned int d = 0; d < dim; ++d)
                    local_errors[1] +=
                      std::pow((grad_analyical_sol_at_qpoints[q_index][d] -
                                grad_solution_at_qpoint[d]),
                               2) *
                      agglo_values.JxW(q_index);
              }
          }
      }

    // Perform reduction and take sqrt of each error
    global_errors[0] = Utilities::MPI::reduce<double>(
      local_errors[0],
      agglomeration_handler.get_triangulation().get_communicator(),
      [](const double a, const double b) { return a + b; });

    global_errors[0] = std::sqrt(global_errors[0]);

    if (compute_semi_H1)
      {
        global_errors[1] = Utilities::MPI::reduce<double>(
          local_errors[1],
          agglomeration_handler.get_triangulation().get_communicator(),
          [](const double a, const double b) { return a + b; });
        global_errors[1] = std::sqrt(global_errors[1]);
      }
  }

  /**
   * Export each polygon in a csv file as a collection of segments.
   */
  template <int dim, int spacedim>
  void
  export_polygon_to_csv_file(
    const AgglomerationHandler<dim, spacedim> &agglomeration_handler,
    const std::string               &filename)
  {
    static_assert(dim == 2); // With 3D, Paraview is much better
    std::ofstream myfile;
    myfile.open(filename + ".csv");

    for (const auto &polytope : agglomeration_handler.polytope_iterators())
      if (polytope->is_locally_owned())
        {
          const std::vector<typename Triangulation<dim, spacedim>::active_face_iterator>
            &boundary = polytope->polytope_boundary();
          for (unsigned int f = 0; f < boundary.size(); ++f)
            {
              myfile << boundary[f]->vertex(0)[0];
              myfile << ",";
              myfile << boundary[f]->vertex(0)[1];
              myfile << ",";
              myfile << boundary[f]->vertex(1)[0];
              myfile << ",";
              myfile << boundary[f]->vertex(1)[1];
              myfile << "\n";
            }
        }


    myfile.close();
  }// namespace internal
}// namespace dealii::PolyUtils

/**
 * The Handler class that stores all the data used in the agglomeration.
 */
template <int dim, int spacedim = dim>
class AgglomerationHandler : public Subscriptor
{
public:
  using agglomeration_iterator = AgglomerationIterator<dim, spacedim>;

  using AgglomerationContainer =
    typename AgglomerationIterator<dim, spacedim>::AgglomerationContainer;

  /**
   * Enumeration for type of cells used in master_slave_relationships
   */
  enum CellAgglomerationType
  {
    master = 0,
    slave  = 1
  };



  explicit AgglomerationHandler(
    const GridTools::Cache<dim, spacedim> &cached_tria);

  AgglomerationHandler() = default;

  ~AgglomerationHandler()
  {
    // disconnect the signal
    tria_listener.disconnect();
  }

  /**
   * Iterator to the first polytope element of const object.
   */
  agglomeration_iterator
  begin() const;

  /**
   * Iterator to the first polytope element.
   */
  agglomeration_iterator
  begin();

  /**
   * Iterator to one after the last polygonal element of const object.
   */
  agglomeration_iterator
  end() const;

  /**
   * Iterator to one after the last polygonal element.
   */
  agglomeration_iterator
  end();

  /**
   * Iterator to the last polygonal element.
   */
  agglomeration_iterator
  last();

  /**
   * Returns an IteratorRange that makes up all the polygonal elements in the
   * mesh.
   */
  IteratorRange<agglomeration_iterator>
  polytope_iterators() const;

  template <int, int>
  friend class AgglomerationIterator;

  template <int, int>
  friend class AgglomerationAccessor;

  /**
   * Distribute degrees of freedom on a grid where some cells have been
   * agglomerated.
   */
  void
  distribute_agglomerated_dofs(const FiniteElement<dim> &fe_space);

  /**
   *
   * Set the degree of the quadrature formula to be used and the proper flags
   * for the FEValues object on the agglomerated cell.
   */
  void
  initialize_fe_values(
    const Quadrature<dim>     &cell_quadrature = QGauss<dim>(1),
    const UpdateFlags         &flags           = UpdateFlags::update_default,
    const Quadrature<dim - 1> &face_quadrature = QGauss<dim - 1>(1),
    const UpdateFlags         &face_flags      = UpdateFlags::update_default);

  /**
   * Given a Triangulation with some agglomerated cells, create the sparsity
   * pattern corresponding to a Discontinuous Galerkin discretization where the
   * agglomerated cells are seen as one **unique** cell, with only the DoFs
   * associated to the master cell of the agglomeration.
   */
  template <typename SparsityPatternType, typename Number = double>
  void
  create_agglomeration_sparsity_pattern(
    SparsityPatternType             &sparsity_pattern,
    const AffineConstraints<Number> &constraints = AffineConstraints<Number>(),
    const bool                       keep_constrained_dofs = true,
    const types::subdomain_id subdomain_id = numbers::invalid_subdomain_id);

  /**
   * Store internally that the given cells are agglomerated. The convenction we
   * take is the following:
   * -1: cell is a master cell
   *
   * @note cells are assumed to be adjacent one to each other, and no check
   * about this is done.
   */
  agglomeration_iterator
  define_agglomerate(const AgglomerationContainer &cells);

  /**
   * Same as above, but checking that every vector of cells is connected. If
   * not, each connected component is agglomerated by part by calling the
   * define_agglomerate() function defined above.
   */
  void
  define_agglomerate_with_check(const AgglomerationContainer &cells);

  /**
   * Returns a constant reference of the Triangulation of the calling agglomeration.
   */
  inline const Triangulation<dim, spacedim> &
  get_triangulation() const;

  /**
   * Returns a constant reference of the FiniteElement of the calling agglomeration.
   */
  inline const FiniteElement<dim, spacedim> &
  get_fe() const;

  /**
   * Returns the default mapping used when initialize the mesh
   */
  inline const Mapping<dim> &
  get_mapping() const;

  /**
   * Returns the mapping of bounding box used in agglomeration
   */
  inline const MappingBox<dim> &
  get_agglomeration_mapping() const;

  /**
   * Return the vector of bounding box of the polytope
   */
  inline const std::vector<BoundingBox<dim>> &
  get_local_bboxes() const;

  /**
   * Return the mesh size of the polytopal mesh. It simply takes the maximum
   * diameter over all the polytopes.
   */
  //double
  //get_mesh_size() const;

  /**
   * Given a master cell, return the index of the related polytope
   */
  inline types::global_cell_index
  cell_to_polytope_index(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const;

  /**
   * Return a map between the pair of index of two adjacent polytopes and the vector of
   * cell and interface id that constructs this interface, which represents a interface between two polytopes
   */
  inline decltype(auto)
  get_interface() const;

  /**
   * Helper function to determine whether or not a cell is a master or a slave
   */
  template <typename CellIterator>
  inline bool
  is_master_cell(const CellIterator &cell) const;

  /**
   * Find (if any) the cells that have the given master index. Note that `idx`
   * is as it can be equal to -1 (meaning that the cell is a master one).
   */
  inline const std::vector<
    typename Triangulation<dim, spacedim>::active_cell_iterator> &
  get_slaves_of_idx(types::global_cell_index idx) const;

  /**
   * Return the vector that stores the correspondence of master and slave cells.
   * The vector has two kinds of elements, if the cell corresponds to the index i is a master cell,
   * the ith vector element is -1; if it's a slave cell, the ith vector element is the
   * index of the master cell of the polytope it belongs to.
   */
  inline const LinearAlgebra::distributed::Vector<float> &
  get_relationships() const;

  /**
   * TODO: remove this in favour of the accessor version.
   *
   * @param master_cell
   * @return std::vector<
   * typename Triangulation<dim, spacedim>::active_cell_iterator>
   */
  inline std::vector<
    typename Triangulation<dim, spacedim>::active_cell_iterator>
  get_agglomerate(
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell) const;

  /**
   * Return a nested map associating a neighboring subdomain id and a pair of ghost 
   * cell id and face index to the shape gradients evaluated at the quadrature points 
   * on that specific interface.
   */
  inline decltype(auto)
  get_recv_gradients() const;

  /**
   * Return a nested map associating a neighboring subdomain id and a pair of ghost 
   * cell id and face index to the shape values evaluated at the quadrature points 
   * on that specific interface.
   */
  inline decltype(auto)
  get_recv_values() const;

  /**
   * Return a nested map associating a neighboring subdomain id and a pair of ghost 
   * cell id and face index to the JxW (Jacobian determinant times quadrature weight) 
   * values evaluated at the quadrature points on that specific interface.
   */
  inline decltype(auto)
  get_recv_jxws() const;

  /**
   * Display the indices of the vector identifying which cell is agglomerated
   * with which master.
   */
  template <class StreamType>
  void
  print_agglomeration(StreamType &out);

  /**
   *
   * Return a constant reference to the DoFHandler underlying the
   * agglomeration. It knows which cell have been agglomerated, and which FE
   * spaces are present on each cell of the triangulation.
   */
  inline const DoFHandler<dim, spacedim> &
  get_dof_handler() const;

  /**
   * Returns the number of agglomerate cells in the grid.
   */
  unsigned int
  n_agglomerates() const;

  /**
   * Return the number of agglomerated faces for a generic deal.II cell.
   */
  //unsigned int
  //n_agglomerated_faces_per_cell(
  //  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  //  const;

  /**
   * Construct a finite element space on the agglomeration.
   */
  const FEValues<dim, spacedim> &
  reinit(const AgglomerationIterator<dim, spacedim> &polytope) const;

  /**
   * For a given polytope and face index, initialize shape functions, normals
   * and quadratures rules to integrate there.
   */
  const FEValuesBase<dim, spacedim> &
  reinit(const AgglomerationIterator<dim, spacedim> &polytope,
         const unsigned int                          face_index) const;

  /**
   *
   * Return a pair of FEValuesBase object reinited from the two component polytopes of the
   * agglomeration.
   */
  std::pair<const FEValuesBase<dim, spacedim> &,
            const FEValuesBase<dim, spacedim> &>
  reinit_interface(const AgglomerationIterator<dim, spacedim> &polytope_in,
                   const AgglomerationIterator<dim, spacedim> &neigh_polytope,
                   const unsigned int                          local_in,
                   const unsigned int local_outside) const;

  /**
   * Return the agglomerated quadrature for the given agglomeration. This
   * amounts to looping over all cells in an agglomeration, collecting together
   * all the rules, and mapping the quadrature points to the reference unit of bounding box.
   */
  Quadrature<dim>
  agglomerated_quadrature(
    const AgglomerationContainer &cells,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &master_cell) const;


  /**
   *
   * This function generalizes the behaviour of cell->face(f)->at_boundary()
   * in the case where f is an index out of the range [0,..., n_faces).
   * In practice, if you call this function with a standard deal.II cell, you
   * have exactly the same result as calling cell->face(f)->at_boundary().
   * Otherwise, if the cell is a master one, it returns a boolean showing 
   * whether that face of the agglomeration is on the boundary or not.
   */
  inline bool
  at_boundary(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              f) const;

  /**
   * Return the number of degrees of freedom per cell.
   */
  inline unsigned int
  n_dofs_per_cell() const noexcept;

  /**
   * Return the number of degrees of the whole mesh
   */
  inline types::global_dof_index
  n_dofs() const noexcept;



  /**
   * Return the collection of boundary faces describing the boundary of the polytope
   * associated to the master cell `cell`. The return type is meant to describe
   * a sequence of edges (in 2D) or faces (in 3D).
   */
  inline const std::vector<typename Triangulation<dim, spacedim>::active_face_iterator> &
  polytope_boundary(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell);


  /**
   * DoFHandler for the agglomerated space
   */
  DoFHandler<dim, spacedim> agglo_dh;

  /**
   * DoFHandler for the finest space: classical deal.II space
   */
  DoFHandler<dim, spacedim> output_dh;

  std::unique_ptr<MappingBox<dim>> box_mapping;

  /**
   * This function stores the information needed to identify which polytopes are
   * ghosted w.r.t the local partition. The issue this function addresses is due
   * to the fact that the layer of ghost cells is made by just one layer of
   * deal.II cells. Therefore, the neighboring polytopes will always be made by
   * some ghost cells and **artificial** ones. This implies that we need to
   * communicate the missing information from the neighboring rank.
   */
  void
  setup_ghost_polytopes();

  /**
   * This function collects all the information needed for discontinuous galerkin method,
   * and exchanges the information with all the interfaces of the neighbor polytope in different MPI ranks.
   * The relevant data is declared in the downside private part, from qpoints to gradients.
   */
  void
  exchange_interface_values();

  /**
   * Given the index of a polytopic element, return a DoFHandler iterator
   * for which DoFs associated to that polytope can be queried.
   */
  inline const typename DoFHandler<dim, spacedim>::active_cell_iterator
  polytope_to_dh_iterator(const types::global_cell_index polytope_index) const;

  /**
   *
   */
  //template <typename RtreeType>
  //void
  //connect_hierarchy(const CellsAgglomerator<dim, RtreeType> &agglomerator);

private:
  /**
   * Initialize connectivity informations
   */
  void
  initialize_agglomeration_data(
    const std::unique_ptr<GridTools::Cache<dim, spacedim>> &cache_tria);

  /**
   * Reinitialize the agglomeration data.
   */
  void
  connect_to_tria_signals()
  {
    // First disconnect existing connections
    tria_listener.disconnect();
    tria_listener = tria->signals.any_change.connect(
      [&]() { this->initialize_agglomeration_data(this->cached_tria); });
  }

  /**
   * Helper function to determine whether or not a cell is a slave cell.
   * Instead of returning a boolean, it gives the index of the master cell. If
   * it's a master cell, then the it returns -1, by construction.
   */

  inline typename Triangulation<dim, spacedim>::active_cell_iterator &
  is_slave_cell_of(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell);

  /**
   * Construct bounding boxes for an agglomeration described by a sequence of
   * cells. This fills also the euler vector
   */
  void
  create_bounding_box(const AgglomerationContainer &polytope);

  /**
   * Return the master index of the polytope this cell belongs to.
   */
  inline types::global_cell_index
  get_master_idx_of_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const;

  /**
   * Returns true if the two given cells are agglomerated together.
   */
  inline bool
  are_cells_agglomerated(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const typename Triangulation<dim, spacedim>::active_cell_iterator
      &other_cell) const;

  /**
   * Assign a finite element index on each cell of a triangulation, depending
   * if it is a master cell, a slave cell, or a standard deal.II cell. A user
   * doesn't need to know the internals of this, the only thing that is
   * relevant is that after the call to the present function, DoFs are
   * distributed in a different way if a cell is a master, slave, or standard
   * cell.
   */
  void
  initialize_hp_structure();


  /**
   * Helper function to call reinit on a master cell.
   */
  const FEValuesBase<dim, spacedim> &
  reinit_master(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              face_number,
    std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
      &agglo_isv_ptr) const;


  /**
   * Helper function to determine whether or not a cell is a slave cell, without
   * any information about his parents.
   */
  template <typename CellIterator>
  inline bool
  is_slave_cell(const CellIterator &cell) const;


  /**
   * Initialize all the necessary connectivity information for an
   * agglomeration. #TODO: loop over polytopes, avoid using explicitly
   * master cells.
   */
  void
  setup_connectivity_of_agglomeration();


  /**
   * Record the number of agglomerations on the grid.
   */
  unsigned int n_agglomerations;


  /**
   * Vector of indices such that v[cell->active_cell_index()] returns
   * { -1 if `cell` is a master cell
   * { `cell_master->active_cell_index()`, i.e. the index of the master cell if
   * `cell` is a slave cell.
   */
  LinearAlgebra::distributed::Vector<float> master_slave_relationships;

  /**
   * Same as the one above, but storing cell iterators rather than indices.
   *
   */
  std::map<types::global_cell_index,
           typename Triangulation<dim, spacedim>::active_cell_iterator>
    master_slave_relationships_iterators;

  using ScratchData = MeshWorker::ScratchData<dim, spacedim>;

  /**
   * Vector that stores how many faces the ith polytope has.
   */
  mutable std::vector<types::global_cell_index> number_of_agglomerated_faces;

  /**
   * Associate a master cell (hence, a given polytope) to its boundary faces.
   * The boundary is described through a vector of face iterators.
   *
   */
  mutable std::map<
    const typename Triangulation<dim, spacedim>::active_cell_iterator,
    std::vector<typename Triangulation<dim>::active_face_iterator>>
    polygon_boundary;


  /**
   * Vector of `BoundingBoxes` s.t. `bboxes[idx]` equals BBOx associated to the
   * agglomeration with master cell indexed by ìdx`. Othwerwise default BBox is
   * empty
   *
   */
  std::vector<BoundingBox<spacedim>> bboxes;

  ////////////////////////////////////////////////////////


  // n_faces
  mutable std::map<types::subdomain_id, std::map<CellId, unsigned int>>
    local_n_faces;

  mutable std::map<types::subdomain_id, std::map<CellId, unsigned int>>
    recv_n_faces;


  // CellId (including slaves)
  mutable std::map<types::subdomain_id, std::map<CellId, CellId>>
    local_cell_ids_neigh_cell;

  mutable std::map<types::subdomain_id, std::map<CellId, CellId>>
    recv_cell_ids_neigh_cell;


  // send to neighborign rank the information that
  // - current polytope id
  // - face f
  // has the following neighboring id.
  mutable std::map<types::subdomain_id,
                   std::map<CellId, std::map<unsigned int, CellId>>>
    local_ghosted_master_id;

  mutable std::map<types::subdomain_id,
                   std::map<CellId, std::map<unsigned int, CellId>>>
    recv_ghosted_master_id;

  // CellIds from neighboring rank
  mutable std::map<types::subdomain_id,
                   std::map<CellId, std::map<unsigned int, bool>>>
    local_bdary_info;

  mutable std::map<types::subdomain_id,
                   std::map<CellId, std::map<unsigned int, bool>>>
    recv_bdary_info;

  // Exchange neighboring bounding boxes
  mutable std::map<types::subdomain_id, std::map<CellId, BoundingBox<dim>>>
    local_ghosted_bbox;

  mutable std::map<types::subdomain_id, std::map<CellId, BoundingBox<dim>>>
    recv_ghosted_bbox;

  // Exchange DoF indices with ghosted polytopes
  mutable std::map<types::subdomain_id,
                   std::map<CellId, std::vector<types::global_dof_index>>>
    local_ghost_dofs;

  mutable std::map<types::subdomain_id,
                   std::map<CellId, std::vector<types::global_dof_index>>>
    recv_ghost_dofs;

  // Exchange qpoints
  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<Point<spacedim>>>>
    local_qpoints;

  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<Point<spacedim>>>>
    recv_qpoints;

  // Exchange jxws
  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<double>>>
    local_jxws;

  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<double>>>
    recv_jxws;

  // Exchange normals
  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<Tensor<1, spacedim>>>>
    local_normals;

  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<Tensor<1, spacedim>>>>
    recv_normals;

  // Exchange values
  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<std::vector<double>>>>
    local_values;

  mutable std::map<
    types::subdomain_id,
    std::map<std::pair<CellId, unsigned int>, std::vector<std::vector<double>>>>
    recv_values;

  // Exchange gradients
  mutable std::map<types::subdomain_id,
                   std::map<std::pair<CellId, unsigned int>,
                            std::vector<std::vector<Tensor<1, spacedim>>>>>
    local_gradients;

  mutable std::map<types::subdomain_id,
                   std::map<std::pair<CellId, unsigned int>,
                            std::vector<std::vector<Tensor<1, spacedim>>>>>
    recv_gradients;

  ////////////////////////////////////////////////////////

  SmartPointer<const Triangulation<dim, spacedim>> tria;

  SmartPointer<const Mapping<dim, spacedim>> mapping;

  std::unique_ptr<GridTools::Cache<dim, spacedim>> cached_tria;

  const MPI_Comm communicator;

  // The FiniteElement space we have on each cell. Currently supported types are
  // FE_DGQ and FE_DGP elements.
  std::unique_ptr<FiniteElement<dim>> fe;

  hp::FECollection<dim, spacedim> fe_collection;

  /**
   * Eulerian vector describing the new cells obtained by the bounding boxes
   */
  LinearAlgebra::distributed::Vector<double> euler_vector;


  /**
   * Use this in reinit(cell) for  (non-agglomerated, standard)  cells,
   * and return the result of scratch.reinit(cell) for cells
   */
  mutable std::unique_ptr<ScratchData> standard_scratch;

  /**
   * Fill this up in reinit(cell), for agglomerated cells, using the custom
   * quadrature, and return the result of
   * scratch.reinit(cell);
   */
  mutable std::unique_ptr<ScratchData> agglomerated_scratch;

  /**
   * Pointer for Immersed Surface Values on non-matching agglomerations.
   * These unique_ptrs cache the FEValues-like objects needed to compute shape 
   * functions and gradients within the polytope, across internal interfaces, 
   * and on physical boundaries.
   */
  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv;

  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv_neigh;

  mutable std::unique_ptr<NonMatching::FEImmersedSurfaceValues<spacedim>>
    agglomerated_isv_bdary;

  /**
   * Listens for mesh modifications to automatically invalidate and rebuild the agglomeration caches.
   */
  boost::signals2::connection tria_listener;

  /**
   * Mandatory internal flags required by the handler for cell evaluation.
   */
  UpdateFlags agglomeration_flags;

  const UpdateFlags internal_agglomeration_flags =
    update_values | update_gradients | update_JxW_values |
    update_quadrature_points;

  /**
   * Update flags for face/interface FE evaluation.
   */
  UpdateFlags agglomeration_face_flags;

  const UpdateFlags internal_agglomeration_face_flags =
    update_quadrature_points | update_normal_vectors | update_values |
    update_gradients | update_JxW_values | update_inverse_jacobians;

  /**
   * Quadrature for current polytope/polytopal face. 
   */
  Quadrature<dim> agglomeration_quad;

  Quadrature<dim - 1> agglomeration_face_quad;

  /**
   * Maps the global active cell index of a master cell to a vector containing 
   * the active cell iterators of all its constituent slave cells.
   */
  std::unordered_map<
    types::global_cell_index,
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
    master2slaves;

  /**
   * Map the master cell index with the polytope index.
   */
  std::map<types::global_cell_index, types::global_cell_index> master2polygon;


  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    master_disconnected;

  // Dummy FiniteElement objects needed only to generate quadratures
  /**
   * Dummy FE_Nothing
   */
  FE_Nothing<dim, spacedim> dummy_fe;

  /**
   * Dummy FEValues, needed for cell quadratures.
   */
  std::unique_ptr<FEValues<dim, spacedim>> no_values;

  /**
   * Dummy FEFaceValues, needed for face quadratures.
   */
  std::unique_ptr<FEFaceValues<dim, spacedim>> no_face_values;

  /**
   * A contiguous container for all of the master cells.
   */
  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    master_cells_container;

  friend class internal::AgglomerationHandlerImplementation<dim, spacedim>;

  internal::PolytopeCache<dim, spacedim> polytope_cache;

  /**
   * Bool that keeps track whether the mesh is composed also by standard deal.II
   * cells as (trivial) polytopes.
   */
  bool hybrid_mesh;

  /*std::map<std::pair<types::global_cell_index, types::global_cell_index>,
           std::vector<types::global_cell_index>>
    parent_child_info;*/

  unsigned int present_extraction_level;
};



// ------------------------------ inline functions -------------------------
template <int dim, int spacedim>
inline const FiniteElement<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::get_fe() const
{
  return *fe;
}

template <int dim, int spacedim>
inline const Mapping<dim> &
AgglomerationHandler<dim, spacedim>::get_mapping() const
{
  return *mapping;
}

template <int dim, int spacedim>
inline const MappingBox<dim> &
AgglomerationHandler<dim, spacedim>::get_agglomeration_mapping() const
{
  return *box_mapping;
}

template <int dim, int spacedim>
inline const Triangulation<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::get_triangulation() const
{
  return *tria;
}

template <int dim, int spacedim>
inline const std::vector<BoundingBox<dim>> &
AgglomerationHandler<dim, spacedim>::get_local_bboxes() const
{
  return bboxes;
}

template <int dim, int spacedim>
inline types::global_cell_index
AgglomerationHandler<dim, spacedim>::cell_to_polytope_index(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const
{
  return master2polygon.at(cell->active_cell_index());
}

template <int dim, int spacedim>
inline decltype(auto)
AgglomerationHandler<dim, spacedim>::get_interface() const
{
  return polytope_cache.interface;
}

template <int dim, int spacedim>
inline const LinearAlgebra::distributed::Vector<float> &
AgglomerationHandler<dim, spacedim>::get_relationships() const
{
  return master_slave_relationships;
}

template <int dim, int spacedim>
inline std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
AgglomerationHandler<dim, spacedim>::get_agglomerate(
  const typename Triangulation<dim, spacedim>::active_cell_iterator
    &master_cell) const
{
  Assert(is_master_cell(master_cell), ExcInternalError());
  auto agglomeration = get_slaves_of_idx(master_cell->active_cell_index());
  agglomeration.push_back(master_cell);
  return agglomeration;
}

template <int dim, int spacedim>
inline decltype(auto)
AgglomerationHandler<dim, spacedim>::get_recv_gradients() const
{
  return (recv_gradients);
}

template <int dim, int spacedim>
inline decltype(auto)
AgglomerationHandler<dim, spacedim>::get_recv_values() const
{
  return (recv_values);
}

template <int dim, int spacedim>
inline decltype(auto)
AgglomerationHandler<dim, spacedim>::get_recv_jxws() const
{
  return (recv_jxws);
}

template <int dim, int spacedim>
template <class StreamType>
void
AgglomerationHandler<dim, spacedim>::print_agglomeration(StreamType &out)
{
  for (const auto &cell : tria->active_cell_iterators())
    out << "Cell with index: " << cell->active_cell_index()
        << " has associated value: "
        << master_slave_relationships[cell->global_active_cell_index()]
        << std::endl;
}

template <int dim, int spacedim>
inline const DoFHandler<dim, spacedim> &
AgglomerationHandler<dim, spacedim>::get_dof_handler() const
{
  return agglo_dh;
}

template <int dim, int spacedim>
inline const std::vector<
  typename Triangulation<dim, spacedim>::active_cell_iterator> &
AgglomerationHandler<dim, spacedim>::get_slaves_of_idx(
  types::global_cell_index idx) const
{
  return master2slaves.at(idx);
}

template <int dim, int spacedim>
template <typename CellIterator>
inline bool
AgglomerationHandler<dim, spacedim>::is_master_cell(
  const CellIterator &cell) const
{
  return master_slave_relationships[cell->global_active_cell_index()] == -1;
}

template <int dim, int spacedim>
template <typename CellIterator>
inline bool
AgglomerationHandler<dim, spacedim>::is_slave_cell(
  const CellIterator &cell) const
{
  return master_slave_relationships[cell->global_active_cell_index()] >= 0;
}

template <int dim, int spacedim>
inline bool
AgglomerationHandler<dim, spacedim>::at_boundary(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int face_index) const
{
  Assert(!is_slave_cell(cell),
         ExcMessage("This function should not be called for a slave cell."));

  return polytope_cache.cell_face_at_boundary
    .at({master2polygon.at(cell->active_cell_index()), face_index})
    .first;
}

template <int dim, int spacedim>
inline unsigned int
AgglomerationHandler<dim, spacedim>::n_dofs_per_cell() const noexcept
{
  return fe->n_dofs_per_cell();
}

template <int dim, int spacedim>
inline types::global_dof_index
AgglomerationHandler<dim, spacedim>::n_dofs() const noexcept
{
  return agglo_dh.n_dofs();
}

template <int dim, int spacedim>
inline const std::vector<typename Triangulation<dim, spacedim>::active_face_iterator> &
AgglomerationHandler<dim, spacedim>::polytope_boundary(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
{
  return polygon_boundary[cell];
}

template <int dim, int spacedim>
inline typename Triangulation<dim, spacedim>::active_cell_iterator &
AgglomerationHandler<dim, spacedim>::is_slave_cell_of(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
{
  return master_slave_relationships_iterators.at(cell->active_cell_index());
}

template <int dim, int spacedim>
inline types::global_cell_index
AgglomerationHandler<dim, spacedim>::get_master_idx_of_cell(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell) const
{
  auto idx = master_slave_relationships[cell->global_active_cell_index()];
  if (idx == -1)
    return cell->global_active_cell_index();
  else
    return static_cast<types::global_cell_index>(idx);
}

template <int dim, int spacedim>
inline bool
AgglomerationHandler<dim, spacedim>::are_cells_agglomerated(
  const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
  const typename Triangulation<dim, spacedim>::active_cell_iterator &other_cell)
  const
{
  // if different subdomain, then **by construction** they will not be together
  // if (cell->subdomain_id() != other_cell->subdomain_id())
  //   return false;
  // else
  return (get_master_idx_of_cell(cell) == get_master_idx_of_cell(other_cell));
}

template <int dim, int spacedim>
inline unsigned int
AgglomerationHandler<dim, spacedim>::n_agglomerates() const
{
  return n_agglomerations;
}

template <int dim, int spacedim>
inline const typename DoFHandler<dim, spacedim>::active_cell_iterator
AgglomerationHandler<dim, spacedim>::polytope_to_dh_iterator(
  const types::global_cell_index polytope_index) const
{
  return master_cells_container[polytope_index]->as_dof_handler_iterator(
    agglo_dh);
}

template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::begin() const
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.begin(), this};
}

template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::begin()
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.begin(), this};
}

template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::end() const
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.end(), this};
}

template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::end()
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {*master_cells_container.end(), this};
}

template <int dim, int spacedim>
AgglomerationIterator<dim, spacedim>
AgglomerationHandler<dim, spacedim>::last()
{
  Assert(n_agglomerations > 0,
         ExcMessage("No agglomeration has been performed."));
  return {master_cells_container.back(), this};
}

template <int dim, int spacedim>
IteratorRange<
  typename AgglomerationHandler<dim, spacedim>::agglomeration_iterator>
AgglomerationHandler<dim, spacedim>::polytope_iterators() const
{
  return IteratorRange<
    typename AgglomerationHandler<dim, spacedim>::agglomeration_iterator>(
    begin(), end());
}

/*template <int dim, int spacedim>
template <typename RtreeType>
void
AgglomerationHandler<dim, spacedim>::connect_hierarchy(
  const CellsAgglomerator<dim, RtreeType> &agglomerator)
{
  parent_child_info        = agglomerator.parent_node_to_children_nodes;
  present_extraction_level = agglomerator.extraction_level;
}*/

#endif
